import os
from glob import glob
import re
import csv

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from matplotlib.widgets import RectangleSelector, RadioButtons, Button
from matplotlib.path import Path

try:
    from spectral.io import envi
except ImportError:
    envi = None


# ============================================================
# PANEL LABELS
# ============================================================

PANEL_LABELS = ("light-gray", "medium-gray", "dark-gray", "black")

PANEL_TITLES = {
    "light-gray": "Light gray panel",
    "medium-gray": "Medium gray panel",
    "dark-gray": "Dark gray panel",
    "black": "Black panel",
}

# These are the expected CSV columns after normalization.
# Your CSV can use:
#   Wavelength, Light Gray, Med. Gray, Dark Gray, Black
CSV_PANEL_COLUMN_ALIASES = {
    "light-gray": ["lightgray"],
    "medium-gray": ["medgray", "mediumgray"],
    "dark-gray": ["darkgray"],
    "black": ["black"],
}


# ============================================================
# ENVI FILE HANDLING
# ============================================================

def is_envi_data_file(path):
    """
    ENVI data files have no extension and matching .hdr file.
    """
    if os.path.isdir(path):
        return False
    if path.lower().endswith(".hdr"):
        return False
    return os.path.exists(path + ".hdr")


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    """
    Natural sorting key:
    raw_2 before raw_10.
    """
    return [tryint(c) for c in re.split(r"([0-9]+)", os.path.basename(s))]


def list_envi_hsi_files(hsi_folder):
    """
    ENVI data files have no extension and matching .hdr file.
    Returns naturally sorted files.
    """
    files = glob(os.path.join(hsi_folder, "*"))
    files = [f for f in files if is_envi_data_file(f)]
    return sorted(files, key=alphanum_key)


def make_output_path(input_path, output_folder, suffix="_rf"):
    base = os.path.basename(input_path)
    return os.path.join(output_folder, base + suffix)


def get_global_coefficient_paths(output_folder, coefficient_basename="global_elm"):
    slope_path = os.path.join(output_folder, coefficient_basename + "_slope.npy")
    intercept_path = os.path.join(output_folder, coefficient_basename + "_intercept.npy")
    return slope_path, intercept_path


def save_global_coefficients(output_folder, slopes, intercepts, coefficient_basename="global_elm"):
    slope_path, intercept_path = get_global_coefficient_paths(
        output_folder,
        coefficient_basename=coefficient_basename,
    )

    np.save(slope_path, slopes.astype(np.float32))
    np.save(intercept_path, intercepts.astype(np.float32))

    print("Saved global ELM coefficients:", flush=True)
    print(slope_path, flush=True)
    print(intercept_path, flush=True)


def load_global_coefficients(output_folder, expected_n_bands=None, coefficient_basename="global_elm"):
    slope_path, intercept_path = get_global_coefficient_paths(
        output_folder,
        coefficient_basename=coefficient_basename,
    )

    slopes = np.load(slope_path).astype(np.float32)
    intercepts = np.load(intercept_path).astype(np.float32)

    if expected_n_bands is not None:
        if len(slopes) != expected_n_bands or len(intercepts) != expected_n_bands:
            raise ValueError(
                f"Global coefficient length mismatch. "
                f"Expected {expected_n_bands} bands, got "
                f"{len(slopes)} slopes and {len(intercepts)} intercepts."
            )

    return slopes, intercepts


def global_coefficients_exist(output_folder, coefficient_basename="global_elm"):
    slope_path, intercept_path = get_global_coefficient_paths(
        output_folder,
        coefficient_basename=coefficient_basename,
    )
    return os.path.isfile(slope_path) and os.path.isfile(intercept_path)


def get_cube_wavelengths(hsi_path):
    """
    Read wavelengths from the original ENVI header.

    Returns:
        wavelengths: np.ndarray, shape [n_bands]
    """
    with rasterio.open(hsi_path) as src:
        n_bands = src.count

    hdr_path = hsi_path + ".hdr"

    if envi is None:
        raise ImportError(
            "spectral is required to read ENVI wavelength metadata. "
            "Install with: pip install spectral"
        )

    if not os.path.exists(hdr_path):
        raise FileNotFoundError(f"ENVI header not found: {hdr_path}")

    meta = envi.read_envi_header(hdr_path)

    if "wavelength" not in meta:
        raise KeyError(f"No wavelength field found in ENVI header: {hdr_path}")

    wavelengths = np.array([float(v) for v in meta["wavelength"]], dtype=np.float32)

    if len(wavelengths) != n_bands:
        raise ValueError(
            f"Wavelength count does not match number of bands for {hsi_path}. "
            f"Found {len(wavelengths)} wavelengths but cube has {n_bands} bands."
        )

    return wavelengths


def save_envi_cube(output_path, cube, reference_profile, wavelengths=None, wavelength_units="Nanometers"):
    """
    Save cube as ENVI headered file with no extension and preserve wavelengths.

    cube shape: [bands, rows, cols]
    """
    profile = reference_profile.copy()
    profile.update(
        {
            "driver": "ENVI",
            "height": cube.shape[1],
            "width": cube.shape[2],
            "count": cube.shape[0],
            "dtype": "float32",
            "interleave": "bsq",
        }
    )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(cube.astype(np.float32))

    if wavelengths is not None:
        if envi is None:
            raise ImportError(
                "spectral is required to write ENVI wavelength metadata. "
                "Install with: pip install spectral"
            )

        hdr_path = output_path + ".hdr"

        if not os.path.exists(hdr_path):
            raise FileNotFoundError(f"Output ENVI header was not created: {hdr_path}")

        metadata = envi.read_envi_header(hdr_path)
        metadata["wavelength"] = [str(float(w)) for w in wavelengths]
        metadata["wavelength units"] = wavelength_units
        envi.write_envi_header(hdr_path, metadata)

    print(f"Saved reflectance cube: {output_path}", flush=True)


# ============================================================
# CSV REFLECTANCE HANDLING
# ============================================================

def normalize_csv_column_name(name):
    """
    Normalize column names such as:
        'Light Gray,' -> 'lightgray'
        'Med. Gray'   -> 'medgray'
        'Dark Gray,'  -> 'darkgray'
    """
    name = str(name).strip().lower()
    name = name.replace(",", "")
    name = name.replace(".", "")
    name = name.replace("_", "")
    name = name.replace("-", "")
    name = name.replace(" ", "")
    return name


def read_panel_reflectance_csv(csv_path, target_wavelengths):
    """
    Read a CSV or tab-delimited file containing panel reflectance spectra.

    Expected logical columns:
        Wavelength
        Light Gray
        Med. Gray
        Dark Gray
        Black

    The function is forgiving about commas/spaces/periods in column names.

    Returns:
        known_reflectance[label] = reflectance spectrum interpolated to target_wavelengths
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Reflectance CSV not found: {csv_path}")

    try:
        import pandas as pd

        df = pd.read_csv(csv_path)

        original_columns = list(df.columns)
        normalized_to_original = {
            normalize_csv_column_name(c): c for c in original_columns
        }

        if "wavelength" not in normalized_to_original:
            raise KeyError(
                f"Could not find a Wavelength column in CSV. "
                f"Found columns: {original_columns}"
            )

        wavelength_col = normalized_to_original["wavelength"]
        csv_wavelengths = df[wavelength_col].to_numpy(dtype=np.float32)

        known_reflectance = {}

        for label, aliases in CSV_PANEL_COLUMN_ALIASES.items():
            found_col = None

            for alias in aliases:
                if alias in normalized_to_original:
                    found_col = normalized_to_original[alias]
                    break

            if found_col is None:
                print(
                    f"Warning: no CSV reflectance column found for {label}. "
                    f"Accepted aliases: {aliases}",
                    flush=True,
                )
                continue

            csv_reflectance = df[found_col].to_numpy(dtype=np.float32)

            order = np.argsort(csv_wavelengths)
            wl_sorted = csv_wavelengths[order]
            rf_sorted = csv_reflectance[order]

            known_reflectance[label] = np.interp(
                target_wavelengths,
                wl_sorted,
                rf_sorted,
                left=rf_sorted[0],
                right=rf_sorted[-1],
            ).astype(np.float32)

    except ImportError:
        known_reflectance = read_panel_reflectance_csv_without_pandas(
            csv_path,
            target_wavelengths,
        )

    if len(known_reflectance) < 2:
        raise RuntimeError(
            "Need at least two panel reflectance spectra from the CSV to fit ELM."
        )

    print("\nLoaded CSV reflectance spectra:")
    for label in known_reflectance:
        print(f"  {label}", flush=True)

    return known_reflectance


def read_panel_reflectance_csv_without_pandas(csv_path, target_wavelengths):
    """
    Fallback CSV reader if pandas is unavailable.
    Works best for comma-separated or tab-separated files with a header.
    """
    with open(csv_path, "r", newline="", errors="ignore") as f:
        sample = f.read(4096)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except Exception:
            dialect = csv.excel

        reader = csv.DictReader(f, dialect=dialect)
        rows = list(reader)

    if len(rows) == 0:
        raise ValueError(f"No data rows found in CSV: {csv_path}")

    original_columns = reader.fieldnames
    normalized_to_original = {
        normalize_csv_column_name(c): c for c in original_columns
    }

    if "wavelength" not in normalized_to_original:
        raise KeyError(
            f"Could not find a Wavelength column in CSV. "
            f"Found columns: {original_columns}"
        )

    wavelength_col = normalized_to_original["wavelength"]
    csv_wavelengths = np.array([float(r[wavelength_col]) for r in rows], dtype=np.float32)

    known_reflectance = {}

    for label, aliases in CSV_PANEL_COLUMN_ALIASES.items():
        found_col = None

        for alias in aliases:
            if alias in normalized_to_original:
                found_col = normalized_to_original[alias]
                break

        if found_col is None:
            print(
                f"Warning: no CSV reflectance column found for {label}. "
                f"Accepted aliases: {aliases}",
                flush=True,
            )
            continue

        csv_reflectance = np.array([float(r[found_col]) for r in rows], dtype=np.float32)

        order = np.argsort(csv_wavelengths)
        wl_sorted = csv_wavelengths[order]
        rf_sorted = csv_reflectance[order]

        known_reflectance[label] = np.interp(
            target_wavelengths,
            wl_sorted,
            rf_sorted,
            left=rf_sorted[0],
            right=rf_sorted[-1],
        ).astype(np.float32)

    return known_reflectance


# ============================================================
# DISPLAY IMAGE
# ============================================================

def stretch_to_uint8(img, nodata_value=0):
    img = img.astype(np.float32)

    valid = np.isfinite(img)
    if nodata_value is not None:
        valid &= img != nodata_value

    out = np.zeros_like(img, dtype=np.float32)

    if valid.sum() == 0:
        return out.astype(np.uint8)

    lo, hi = np.nanpercentile(img[valid], [2, 98])
    out[valid] = np.clip((img[valid] - lo) / (hi - lo + 1e-8), 0, 1)

    return (out * 255).astype(np.uint8)


def make_rgb_preview(cube, rgb_bands=(110, 60, 30), nodata_value=0):
    rgb = np.zeros((cube.shape[1], cube.shape[2], 3), dtype=np.uint8)

    for i, b in enumerate(rgb_bands):
        rgb[:, :, i] = stretch_to_uint8(cube[b], nodata_value=nodata_value)

    return rgb


# ============================================================
# RECTANGLE COLLECTION GUI
# ============================================================

class PanelPolygonCollector:
    """
    Matplotlib GUI:
    - Pick panel label using radio buttons.
    - Draw rectangle around the selected calibration panel.
    - Undo, Submit, Skip.

    Each saved rectangle contains:
        {
            "label": label,
            "vertices": verts,
        }
    """

    def __init__(self, image, title=""):
        self.image = image
        self.title = title

        self.polygons = []
        self.skipped = False
        self.done = False

        self.current_label = "light-gray"
        self.selector = None
        self.draw_mode = False

        # Leave room on right side for controls.
        self.fig, self.ax = plt.subplots(figsize=(13, 8))

        # Leave room on right side for controls.
        plt.subplots_adjust(left=0.04, right=0.74, bottom=0.06, top=0.92)

        self.ax.imshow(image)
        self.ax.set_title(title)
        self.ax.set_axis_off()

        # Clean compact instruction/status text.
        # Keep this ABOVE the buttons so it does not overlap.
        self.guide_text = self.fig.text(
            0.77,
            0.92,
            "",
            fontsize=11,
            va="top",
            ha="left",
        )

        # Panel label selector
        radio_ax = plt.axes([0.77, 0.66, 0.19, 0.18])
        self.radio = RadioButtons(radio_ax, PANEL_LABELS)
        self.radio.on_clicked(self.set_label)

        # Draw button
        draw_ax = plt.axes([0.77, 0.55, 0.19, 0.06])

        # Management buttons with more vertical separation
        undo_ax = plt.axes([0.77, 0.36, 0.19, 0.06])
        submit_ax = plt.axes([0.77, 0.27, 0.19, 0.06])
        skip_ax = plt.axes([0.77, 0.18, 0.19, 0.06])

        self.draw_button = Button(draw_ax, "Draw Rect")
        self.undo_button = Button(undo_ax, "Undo Last")
        self.submit_button = Button(submit_ax, "Submit")
        self.skip_button = Button(skip_ax, "Skip Image")

        self.draw_button.on_clicked(self.enable_draw_mode)
        self.undo_button.on_clicked(self.undo)
        self.submit_button.on_clicked(self.submit)
        self.skip_button.on_clicked(self.skip)

        self.update_guide_text()

    def update_guide_text(self):
        if self.draw_mode:
            status = "Drag rectangle on image"
        else:
            status = "Pick panel type, then Draw Rect"

        self.guide_text.set_text(""
        )

        if plt.fignum_exists(self.fig.number):
            self.fig.canvas.draw_idle()

    def set_label(self, label):
        self.current_label = label
        self.disable_rectangle_selector()
        self.draw_mode = False

        print(f"Selected panel label: {self.current_label}", flush=True)
        self.update_guide_text()

    def turn_off_toolbar_modes(self):
        """
        Turn off Matplotlib toolbar pan/zoom modes before enabling RectangleSelector.

        This is important because if pan or zoom remains active, mouse drag
        events are captured by the toolbar instead of the rectangle selector.
        """
        toolbar = self.fig.canvas.toolbar

        if toolbar is None:
            return

        try:
            mode = str(toolbar.mode).lower()

            if "pan" in mode:
                toolbar.pan()

            if "zoom" in mode:
                toolbar.zoom()

        except Exception:
            pass

    def disable_rectangle_selector(self):
        if self.selector is not None:
            try:
                self.selector.set_active(False)
                self.selector.disconnect_events()
            except Exception:
                pass
            self.selector = None

    def enable_draw_mode(self, event=None):
        self.draw_mode = True

        self.turn_off_toolbar_modes()
        self.disable_rectangle_selector()

        self.selector = RectangleSelector(
            self.ax,
            self.on_rectangle,
            useblit=False,
            button=[1],
            minspanx=2,
            minspany=2,
            spancoords="pixels",
            interactive=True,
            props=dict(edgecolor="yellow", linewidth=2, fill=False, alpha=0.9),
        )

        self.selector.set_active(True)

        print(f"Draw mode enabled for panel: {self.current_label}", flush=True)
        self.update_guide_text()

    def on_rectangle(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        if x1 is None or y1 is None or x2 is None or y2 is None:
            return

        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        if abs(xmax - xmin) < 2 or abs(ymax - ymin) < 2:
            print("Rectangle too small; not saved.", flush=True)
            return

        verts = np.array(
            [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
            ],
            dtype=np.float32,
        )

        label = self.current_label

        self.polygons.append(
            {
                "label": label,
                "vertices": verts,
            }
        )

        self.draw_rectangle_on_axes(verts, label)

        print(
            f"Saved rectangle {len(self.polygons)}: label={label}",
            flush=True,
        )

        self.draw_mode = False
        self.disable_rectangle_selector()

        if plt.fignum_exists(self.fig.number):
            self.fig.canvas.draw_idle()

        self.update_guide_text()

    def draw_rectangle_on_axes(self, verts, label):
        closed = np.vstack([verts, verts[0]])
        self.ax.plot(closed[:, 0], closed[:, 1], linewidth=2, color="yellow")

        cx, cy = verts.mean(axis=0)

        self.ax.text(
            cx,
            cy,
            label,
            color="red",
            fontsize=12,
            weight="bold",
            ha="center",
            va="center",
        )

    def redraw(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.ax.clear()
        self.ax.imshow(self.image)
        self.ax.set_title(self.title)
        self.ax.set_axis_off()

        for poly in self.polygons:
            self.draw_rectangle_on_axes(
                poly["vertices"],
                poly["label"],
            )

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        if plt.fignum_exists(self.fig.number):
            self.fig.canvas.draw_idle()

    def undo(self, event=None):
        if len(self.polygons) == 0:
            print("Nothing to undo.", flush=True)
            return

        removed = self.polygons.pop()
        print(f"Removed last rectangle: label={removed.get('label')}", flush=True)

        self.disable_rectangle_selector()
        self.draw_mode = False
        self.redraw()
        self.update_guide_text()

    def _finish(self):
        self.done = True
        self.disable_rectangle_selector()

        try:
            self.fig.canvas.stop_event_loop()
        except Exception:
            pass

        plt.close(self.fig)

    def submit(self, event=None):
        print(f"Submit clicked: {len(self.polygons)} rectangles collected.", flush=True)
        self._finish()

    def skip(self, event=None):
        print("Skip clicked.", flush=True)
        self.skipped = True
        self._finish()

    def run(self):
        plt.show(block=True)
        print("Rectangle GUI closed. Returning to main code.", flush=True)
        return self.polygons, self.skipped


def collect_polygons_for_cube(rgb_image, basename):
    collector = PanelPolygonCollector(
        rgb_image,
        title=f"Draw calibration panel rectangles: {basename}",
    )
    return collector.run()


# ============================================================
# SPECTRAL EXTRACTION
# ============================================================

def polygon_to_mask(vertices, shape):
    rows, cols = shape

    x = vertices[:, 0]
    y = vertices[:, 1]

    xmin = max(int(np.floor(np.min(x))), 0)
    xmax = min(int(np.ceil(np.max(x))) + 1, cols)

    ymin = max(int(np.floor(np.min(y))), 0)
    ymax = min(int(np.ceil(np.max(y))) + 1, rows)

    mask = np.zeros((rows, cols), dtype=bool)

    if xmax <= xmin or ymax <= ymin:
        return mask

    yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
    points = np.column_stack([xx.ravel(), yy.ravel()])

    path = Path(vertices)
    local_mask = path.contains_points(points).reshape(ymax - ymin, xmax - xmin)

    mask[ymin:ymax, xmin:xmax] = local_mask

    return mask


def extract_polygon_spectra(cube, polygons, nodata_value=0):
    """
    Extract mean radiance spectra from selected panel rectangles.

    Returns:
        panel_samples[label] = list of measured radiance spectra
    """
    panel_samples = {label: [] for label in PANEL_LABELS}

    rows, cols = cube.shape[1:]

    for poly in polygons:
        label = poly["label"]
        verts = poly["vertices"]

        if label not in panel_samples:
            print(f"Warning: unknown panel label {label}; skipping.")
            continue

        mask = polygon_to_mask(verts, (rows, cols))
        pixels = cube[:, mask]

        if pixels.shape[1] == 0:
            print(f"Warning: rectangle {label} had no pixels.")
            continue

        valid = np.all(np.isfinite(pixels), axis=0)
        if nodata_value is not None:
            valid &= np.all(pixels != nodata_value, axis=0)

        if valid.sum() == 0:
            print(f"Warning: rectangle {label} had no valid pixels.")
            continue

        spectrum = np.nanmean(pixels[:, valid], axis=1)
        panel_samples[label].append(spectrum.astype(np.float32))

    return panel_samples


def merge_panel_samples(global_panel_samples, new_panel_samples):
    for label in PANEL_LABELS:
        global_panel_samples[label].extend(new_panel_samples.get(label, []))


def average_panel_radiance_spectra(global_panel_samples):
    """
    Average all collected radiance spectra for each panel type across all cubes.

    Returns:
        averaged_radiance[label] = average measured radiance spectrum
    """
    averaged_radiance = {}

    print("\nAveraging collected panel radiance spectra")
    print("=" * 80)

    for label in PANEL_LABELS:
        spectra = global_panel_samples.get(label, [])

        if len(spectra) == 0:
            print(f"{label}: no samples collected.", flush=True)
            continue

        stacked = np.stack(spectra, axis=0)
        averaged_radiance[label] = np.nanmean(stacked, axis=0).astype(np.float32)

        print(f"{label}: averaged {len(spectra)} spectra.", flush=True)

    if len(averaged_radiance) < 2:
        raise RuntimeError(
            "Need at least two panel types with collected radiance spectra to fit ELM."
        )

    return averaged_radiance


# ============================================================
# PANEL SPECTRA PLOT
# ============================================================

def init_panel_spectra_figure(ylabel="Radiance"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    panel_order = ["light-gray", "medium-gray", "dark-gray", "black"]

    for ax, label in zip(axes, panel_order):
        ax.set_title(PANEL_TITLES[label])
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.ion()
    fig.show()

    return fig, dict(zip(panel_order, axes))


def plot_average_panel_radiance(wavelengths, averaged_radiance):
    fig, axes = init_panel_spectra_figure(ylabel="Average radiance")

    for label, spectrum in averaged_radiance.items():
        ax = axes[label]
        ax.plot(wavelengths, spectrum, label=f"{label} average")
        ax.legend(fontsize=8)

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show(block=False)
    plt.pause(0.5)


# ============================================================
# ELM FIT + APPLY
# ============================================================

def fit_global_elm_coefficients_from_averaged_panels(
    averaged_radiance,
    known_reflectance,
):
    """
    Fit one global ELM coefficient set from averaged panel radiance spectra.

    For each spectral band:
        reflectance = slope * radiance + intercept

    The same slopes/intercepts are applied to all cubes.
    """
    common_labels = [
        label for label in PANEL_LABELS
        if label in averaged_radiance and label in known_reflectance
    ]

    if len(common_labels) < 2:
        raise RuntimeError(
            "Need at least two matching panel types between collected radiance "
            "and CSV reflectance to fit ELM."
        )

    print("\nFitting global ELM using panel types:")
    for label in common_labels:
        print(f"  {label}", flush=True)

    X = np.stack([averaged_radiance[label] for label in common_labels], axis=0)
    Y = np.stack([known_reflectance[label] for label in common_labels], axis=0)

    if X.shape != Y.shape:
        raise ValueError(
            f"Radiance and reflectance arrays do not have the same shape. "
            f"Radiance: {X.shape}, Reflectance: {Y.shape}"
        )

    n_bands = X.shape[1]
    slopes = np.zeros(n_bands, dtype=np.float32)
    intercepts = np.zeros(n_bands, dtype=np.float32)

    for b in range(n_bands):
        xb = X[:, b]
        yb = Y[:, b]

        valid = np.isfinite(xb) & np.isfinite(yb)

        if valid.sum() < 2:
            slopes[b] = 0.0
            intercepts[b] = np.nan
            continue

        xb = xb[valid]
        yb = yb[valid]

        if np.nanstd(xb) < 1e-12:
            slopes[b] = 0.0
            intercepts[b] = np.nanmean(yb)
        else:
            slope, intercept = np.polyfit(xb, yb, deg=1)
            slopes[b] = slope
            intercepts[b] = intercept

    return slopes, intercepts


def apply_elm(cube, slopes, intercepts, nodata_value=0, clip=True):
    out = cube.astype(np.float32).copy()

    for b in range(cube.shape[0]):
        band = cube[b].astype(np.float32)

        valid = np.isfinite(band)
        if nodata_value is not None:
            valid &= band != nodata_value

        calibrated = slopes[b] * band + intercepts[b]

        out[b] = calibrated
        out[b, ~valid] = nodata_value

    return out.astype(np.float32)


# ============================================================
# MAIN WORKFLOW
# ============================================================

def run_interactive_global_elm_calibration(
    hsi_folder,
    output_folder,
    reflectance_csv_path,
    rgb_bands=(110, 60, 30),
    nodata_value=0,
    use_existing_global_coefficients=False,
    collect_panels_for_all_images=True,
    overwrite_reflectance_outputs=True,
    coefficient_basename="global_elm",
):
    """
    Global ELM calibration workflow.

    New behavior:
        1. User draws panel rectangles only. No .sig file picking.
        2. Radiance spectra from matching panel types are averaged across cubes.
        3. Averaged radiance spectra are matched to CSV reflectance spectra.
        4. One global ELM coefficient set is fit.
        5. The same coefficient set is applied to all cubes.

    CSV file must contain columns equivalent to:
        Wavelength, Light Gray, Med. Gray, Dark Gray, Black
    """
    os.makedirs(output_folder, exist_ok=True)

    hsi_files = list_envi_hsi_files(hsi_folder)
    all_hsi_basenames = [os.path.basename(f) for f in hsi_files]

    if len(hsi_files) == 0:
        raise RuntimeError(f"No ENVI HSI files found in: {hsi_folder}")

    print(f"Found {len(hsi_files)} HSI cubes.", flush=True)

    # Use the first cube wavelengths as the target wavelength grid.
    target_wavelengths = get_cube_wavelengths(hsi_files[0])
    target_n_bands = len(target_wavelengths)

    # Check all cubes have compatible wavelength grids.
    for hsi_path in hsi_files:
        wavelengths = get_cube_wavelengths(hsi_path)

        if len(wavelengths) != target_n_bands:
            raise ValueError(
                f"Wavelength count mismatch for {hsi_path}. "
                f"Expected {target_n_bands}, got {len(wavelengths)}."
            )

        if not np.allclose(wavelengths, target_wavelengths):
            print(
                f"Warning: wavelength grid differs for {os.path.basename(hsi_path)}. "
                f"Radiance panel spectra from this cube will be interpolated "
                f"to the first cube wavelength grid before averaging.",
                flush=True,
            )

    if use_existing_global_coefficients and global_coefficients_exist(
        output_folder,
        coefficient_basename=coefficient_basename,
    ):
        print("\nUsing existing global ELM coefficients.", flush=True)

        slopes, intercepts = load_global_coefficients(
            output_folder,
            expected_n_bands=target_n_bands,
            coefficient_basename=coefficient_basename,
        )

    else:
        known_reflectance = read_panel_reflectance_csv(
            reflectance_csv_path,
            target_wavelengths,
        )

        global_panel_samples = {label: [] for label in PANEL_LABELS}

        if collect_panels_for_all_images:
            basenames_to_collect = all_hsi_basenames
        else:
            # With global coefficients, collecting from all images is usually best.
            # This branch is kept for flexibility but defaults to all images.
            basenames_to_collect = all_hsi_basenames

        hsi_lookup = {os.path.basename(p): p for p in hsi_files}

        print("\nImages for panel collection:")
        for b in basenames_to_collect:
            print(f"  {b}", flush=True)

        for basename in basenames_to_collect:
            hsi_path = hsi_lookup[basename]

            print("\n" + "=" * 80)
            print(f"Loading: {basename}", flush=True)

            with rasterio.open(hsi_path) as src:
                cube = src.read().astype(np.float32)

            wavelengths = get_cube_wavelengths(hsi_path)

            rgb = make_rgb_preview(
                cube,
                rgb_bands=rgb_bands,
                nodata_value=nodata_value,
            )

            print("Opening rectangle GUI...", flush=True)
            polygons, skipped = collect_polygons_for_cube(rgb, basename)
            print("Returned from rectangle GUI.", flush=True)

            plt.pause(0.2)

            if skipped:
                print(f"Skipped panel collection for: {basename}", flush=True)
                continue

            print(f"Extracting spectra from {len(polygons)} rectangles...", flush=True)
            panel_samples = extract_polygon_spectra(
                cube,
                polygons,
                nodata_value=nodata_value,
            )
            print("Finished extracting rectangle spectra.", flush=True)

            # If this cube has a slightly different wavelength grid,
            # interpolate its collected radiance spectra to the target grid
            # before merging into the global pool.
            if (
                len(wavelengths) != len(target_wavelengths)
                or not np.allclose(wavelengths, target_wavelengths)
            ):
                for label in PANEL_LABELS:
                    interpolated = []

                    for spectrum in panel_samples[label]:
                        interpolated_spectrum = np.interp(
                            target_wavelengths,
                            wavelengths,
                            spectrum,
                            left=spectrum[0],
                            right=spectrum[-1],
                        ).astype(np.float32)
                        interpolated.append(interpolated_spectrum)

                    panel_samples[label] = interpolated

            merge_panel_samples(global_panel_samples, panel_samples)

        averaged_radiance = average_panel_radiance_spectra(global_panel_samples)

        plot_average_panel_radiance(
            target_wavelengths,
            averaged_radiance,
        )

        slopes, intercepts = fit_global_elm_coefficients_from_averaged_panels(
            averaged_radiance=averaged_radiance,
            known_reflectance=known_reflectance,
        )

        save_global_coefficients(
            output_folder,
            slopes,
            intercepts,
            coefficient_basename=coefficient_basename,
        )

    # --------------------------------------------------------
    # Apply one global coefficient set to every cube
    # --------------------------------------------------------
    print("\nApplying global ELM coefficients to all cubes")
    print("=" * 80)

    for hsi_path in hsi_files:
        basename = os.path.basename(hsi_path)
        output_path = make_output_path(hsi_path, output_folder, suffix="_rf")

        print("\n" + "=" * 80)
        print(f"Calibrating: {basename}", flush=True)

        if (
            not overwrite_reflectance_outputs
            and os.path.exists(output_path)
            and os.path.exists(output_path + ".hdr")
        ):
            print(f"Reflectance output already exists; skipping: {output_path}", flush=True)
            continue

        wavelengths = get_cube_wavelengths(hsi_path)

        with rasterio.open(hsi_path) as src:
            cube = src.read().astype(np.float32)
            profile = src.profile.copy()
            n_bands = src.count

        if n_bands != len(slopes):
            raise ValueError(
                f"Band count mismatch for {basename}. "
                f"Cube has {n_bands} bands but coefficients have {len(slopes)} bands."
            )

        rf_cube = apply_elm(
            cube,
            slopes,
            intercepts,
            nodata_value=nodata_value,
            clip=True,
        )

        save_envi_cube(
            output_path,
            rf_cube,
            profile,
            wavelengths=wavelengths,
            wavelength_units="Nanometers",
        )

    print("\nDone.", flush=True)



# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    run_interactive_global_elm_calibration(
        hsi_folder="/mnt/data-drive/projects_data/darpa/beets/2022/3rd_20220726/nano/1502_radiance_coreg",
        output_folder="/mnt/data-drive/projects_data/darpa/beets/2022/3rd_20220726/nano/1502_radiance_coreg_ref_global",

        # CSV columns should be equivalent to:
        # Wavelength, Light Gray, Med. Gray, Dark Gray, Black
        reflectance_csv_path="/mnt/data-drive/projects_data/darpa/beets/2022/4th_20220810/SVC/2021_reflectance_calibration.csv",

        rgb_bands=(110, 60, 30),
        nodata_value=0,

        # Set True only if you already created global_elm_slope.npy
        # and global_elm_intercept.npy in the output folder.
        use_existing_global_coefficients=True,

        # For global calibration, collecting panels from all cubes is usually best.
        collect_panels_for_all_images=True,

        overwrite_reflectance_outputs=True,
        coefficient_basename="global_elm",
    )