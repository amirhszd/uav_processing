import os
from glob import glob
import re

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


def get_coefficient_paths(output_path):
    slope_path = output_path + "_elm_slope.npy"
    intercept_path = output_path + "_elm_intercept.npy"
    return slope_path, intercept_path


def coefficients_exist(output_path):
    slope_path, intercept_path = get_coefficient_paths(output_path)
    return os.path.isfile(slope_path) and os.path.isfile(intercept_path)


def load_coefficients(output_path, expected_n_bands=None):
    slope_path, intercept_path = get_coefficient_paths(output_path)

    slopes = np.load(slope_path).astype(np.float32)
    intercepts = np.load(intercept_path).astype(np.float32)

    if expected_n_bands is not None:
        if len(slopes) != expected_n_bands or len(intercepts) != expected_n_bands:
            raise ValueError(
                f"Coefficient length mismatch for {output_path}. "
                f"Expected {expected_n_bands} bands, got "
                f"{len(slopes)} slopes and {len(intercepts)} intercepts."
            )

    return slopes, intercepts


def save_coefficients(output_path, slopes, intercepts):
    slope_path, intercept_path = get_coefficient_paths(output_path)
    np.save(slope_path, slopes.astype(np.float32))
    np.save(intercept_path, intercepts.astype(np.float32))

    print("Saved coefficients:", flush=True)
    print(slope_path, flush=True)
    print(intercept_path, flush=True)


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
# REFLECTANCE .SIG FILES
# ============================================================

def read_sig_reflectance(sig_path):
    """
    Reads a .sig-like file where:
        first numeric column = wavelength
        last numeric column = reflectance.
    """
    rows = []

    with open(sig_path, "r", errors="ignore") as f:
        for line in f:
            parts = line.replace(",", " ").split()

            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except ValueError:
                    pass

            if len(nums) >= 2:
                rows.append(nums)

    if len(rows) == 0:
        raise ValueError(f"No numeric wavelength/reflectance rows found in {sig_path}")

    wavelengths = np.array([r[0] for r in rows], dtype=np.float32)
    reflectance = np.array([r[-1] for r in rows], dtype=np.float32)

    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    reflectance = reflectance[order]

    return wavelengths, reflectance


def interpolate_reflectance_files(reflectance_file_dict, target_wavelengths):
    out = {}

    for label, sig_path in reflectance_file_dict.items():
        if sig_path is None:
            continue

        wl, rf = read_sig_reflectance(sig_path)

        out[label] = np.interp(
            target_wavelengths,
            wl,
            rf,
            left=rf[0],
            right=rf[-1],
        ).astype(np.float32)

    return out


def validate_reflectance_files(
    reflectance_files,
    required_labels=("black", "dark-gray", "medium-gray", "light-gray"),
):
    missing = []

    for label in required_labels:
        path = reflectance_files.get(label)

        if path is None:
            missing.append((label, "None"))
        elif not os.path.isfile(path):
            missing.append((label, path))

    if missing:
        msg = ["Missing reflectance files:"]
        for label, path in missing:
            msg.append(f"  {label}: {path}")
        raise FileNotFoundError("\n".join(msg))

    print("All reflectance files found:")
    for label in required_labels:
        print(f"  {label}: {reflectance_files[label]}", flush=True)


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
    - Use Zoom/Pan button to zoom normally.
    - Use Draw Rectangle button to draw a rectangular panel ROI.
    - Pick current panel label using radio buttons.
    - Pick the matching .sig field calibration file before drawing.
    - Undo, Submit, Skip.

    Each saved rectangle now contains:
        {
            "label": label,
            "vertices": verts,
            "sig_path": selected_sig_path,
        }

    This is important because the user chooses the matching field calibration
    file while still seeing the panel location in the image.
    """

    def __init__(self, image, title="", svc_folder=None):
        self.image = image
        self.title = title
        self.svc_folder = svc_folder

        self.polygons = []
        self.skipped = False
        self.done = False
        self.current_label = "light-gray"
        self.selector = None
        self.draw_mode = False

        self.selected_sig_path = None

        if self.svc_folder is not None:
            self.sig_files = list_sig_files(self.svc_folder)
        else:
            self.sig_files = []

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.05, right=0.78, bottom=0.08, top=0.92)

        self.ax.imshow(image)
        self.ax.set_title(title)
        self.ax.set_axis_off()

        radio_ax = plt.axes([0.81, 0.64, 0.15, 0.20])
        self.radio = RadioButtons(radio_ax, PANEL_LABELS)
        self.radio.on_clicked(self.set_label)

        pick_sig_ax = plt.axes([0.81, 0.55, 0.15, 0.06])
        draw_ax = plt.axes([0.81, 0.47, 0.15, 0.06])
        zoom_ax = plt.axes([0.81, 0.39, 0.15, 0.06])
        undo_ax = plt.axes([0.81, 0.31, 0.15, 0.06])
        submit_ax = plt.axes([0.81, 0.22, 0.15, 0.06])
        skip_ax = plt.axes([0.81, 0.14, 0.15, 0.06])

        self.pick_sig_button = Button(pick_sig_ax, "Pick SIG")
        self.draw_button = Button(draw_ax, "Draw Rect")
        self.zoom_button = Button(zoom_ax, "Zoom/Pan")
        self.undo_button = Button(undo_ax, "Undo")
        self.submit_button = Button(submit_ax, "Submit")
        self.skip_button = Button(skip_ax, "Skip")

        self.pick_sig_button.on_clicked(self.pick_sig_file)
        self.draw_button.on_clicked(self.enable_draw_mode)
        self.zoom_button.on_clicked(self.enable_zoom_mode)
        self.undo_button.on_clicked(self.undo)
        self.submit_button.on_clicked(self.submit)
        self.skip_button.on_clicked(self.skip)

        self.text = self.fig.text(
            0.81,
            0.03,
            "",
            fontsize=9,
        )

        self.enable_zoom_mode()

    def selected_sig_name(self):
        if self.selected_sig_path is None:
            return "None"
        return os.path.basename(self.selected_sig_path)

    def update_status_text(self):
        mode = "Draw Rect" if self.draw_mode else "Zoom/Pan"

        self.text.set_text(
            f"Mode: {mode}\n"
            f"Panel: {self.current_label}\n"
            f"SIG: {self.selected_sig_name()}\n\n"
            f"Pick label + SIG,\nthen draw rectangle."
        )

        if plt.fignum_exists(self.fig.number):
            self.fig.canvas.draw_idle()

    def set_label(self, label):
        self.current_label = label
        self.update_status_text()

    def pick_sig_file(self, event=None):
        """
        Open a file picker to choose the .sig file for the next rectangle.

        This works inside the same GUI workflow. The user can look at the
        image, identify the panel/field location, choose the correct .sig,
        and then draw the rectangle.
        """
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        initialdir = self.svc_folder if self.svc_folder is not None else os.getcwd()

        path = filedialog.askopenfilename(
            parent=root,
            initialdir=initialdir,
            title=f"Choose .sig file for {self.current_label}",
            filetypes=[
                ("SVC / SIG files", "*.sig"),
                ("All files", "*.*"),
            ],
        )

        root.destroy()

        if path:
            self.selected_sig_path = path
            print(
                f"Selected SIG for {self.current_label}: {self.selected_sig_path}",
                flush=True,
            )

        self.update_status_text()

    def turn_off_toolbar_modes(self):
        """
        Turn off Matplotlib toolbar pan/zoom modes before enabling RectangleSelector.

        This is important because if pan or zoom remains active, mouse drag events
        are captured by the toolbar instead of the rectangle selector.
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

    def enable_draw_mode(self, event=None):
        if self.selected_sig_path is None:
            print(
                "Please click Pick SIG and choose a field calibration file before drawing.",
                flush=True,
            )
            self.update_status_text()
            return

        self.draw_mode = True

        # Important: disable toolbar pan/zoom before enabling rectangle drawing.
        # Otherwise drag events may be captured by the Matplotlib toolbar.
        self.turn_off_toolbar_modes()

        if self.selector is not None:
            try:
                self.selector.set_active(False)
                self.selector.disconnect_events()
            except Exception:
                pass
            self.selector = None

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

        print(
            f"Draw mode enabled. Panel={self.current_label}, SIG={self.selected_sig_name()}",
            flush=True,
        )

        self.update_status_text()

    def enable_zoom_mode(self, event=None):
        self.draw_mode = False

        if self.selector is not None:
            try:
                self.selector.set_active(False)
                self.selector.disconnect_events()
            except Exception:
                pass
            self.selector = None

        print("Zoom/Pan mode enabled. Use the Matplotlib toolbar if needed.", flush=True)

        self.update_status_text()

    def on_rectangle(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        if x1 is None or y1 is None or x2 is None or y2 is None:
            return

        if self.selected_sig_path is None:
            print("No .sig file selected. Rectangle was not saved.", flush=True)
            return

        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        if abs(xmax - xmin) < 2 or abs(ymax - ymin) < 2:
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
        sig_path = self.selected_sig_path

        self.polygons.append(
            {
                "label": label,
                "vertices": verts,
                "sig_path": sig_path,
            }
        )

        self.draw_rectangle_on_axes(verts, label, sig_path)

        # Force the user to intentionally choose the .sig file for the next ROI.
        # This prevents accidentally reusing the wrong field calibration file.
        self.selected_sig_path = None

        if plt.fignum_exists(self.fig.number):
            self.fig.canvas.draw_idle()

        self.enable_draw_mode()

    def draw_rectangle_on_axes(self, verts, label, sig_path=None):
        closed = np.vstack([verts, verts[0]])
        self.ax.plot(closed[:, 0], closed[:, 1], linewidth=2, color="yellow")

        cx, cy = verts.mean(axis=0)

        sig_name = os.path.basename(sig_path) if sig_path is not None else "no SIG"
        display_text = f"{label}\n{sig_name}"

        self.ax.text(
            cx,
            cy,
            display_text,
            color="red",
            fontsize=10,
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
                poly.get("sig_path"),
            )

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        if plt.fignum_exists(self.fig.number):
            self.fig.canvas.draw_idle()

    def undo(self, event=None):
        if len(self.polygons) == 0:
            return

        self.polygons.pop()
        self.redraw()

        if self.draw_mode:
            self.enable_draw_mode()

    def _finish(self):
        self.done = True

        if self.selector is not None:
            try:
                self.selector.disconnect_events()
            except Exception:
                pass
            self.selector = None

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


def collect_polygons_for_cube(rgb_image, basename, svc_folder=None):
    collector = PanelPolygonCollector(
        rgb_image,
        title=f"Draw calibration panel rectangles: {basename}",
        svc_folder=svc_folder,
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
    Extract mean spectra from selected panel rectangles.

    Returns:
        panel_samples[label] = list of dicts

        Example:
            panel_samples["light-gray"] = [
                {
                    "spectrum": measured_radiance_spectrum,
                    "sig_path": matching_field_reflectance_file,
                }
            ]
    """
    panel_samples = {label: [] for label in PANEL_LABELS}

    rows, cols = cube.shape[1:]

    for poly in polygons:
        label = poly["label"]
        verts = poly["vertices"]
        sig_path = poly.get("sig_path")

        if sig_path is None:
            print(f"Warning: rectangle {label} has no .sig file; skipping.")
            continue

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

        panel_samples[label].append(
            {
                "spectrum": spectrum.astype(np.float32),
                "sig_path": sig_path,
            }
        )

    return panel_samples


# ============================================================
# CALIBRATION SETS
# ============================================================

def make_panel_sets(panel_samples, basename):
    available = [k for k, v in panel_samples.items() if len(v) > 0]

    sets = []
    used = set()

    if "black" in available and "dark-gray" in available:
        sets.append(
            {
                "id": f"S{len(sets)+1}_{basename}",
                "labels": ["black", "dark-gray"],
            }
        )
        used.update(["black", "dark-gray"])

    if "light-gray" in available:
        sets.append(
            {
                "id": f"S{len(sets)+1}_{basename}",
                "labels": ["light-gray"],
            }
        )
        used.add("light-gray")

    if "medium-gray" in available:
        sets.append(
            {
                "id": f"S{len(sets)+1}_{basename}",
                "labels": ["medium-gray"],
            }
        )
        used.add("medium-gray")

    for label in available:
        if label not in used:
            sets.append(
                {
                    "id": f"S{len(sets)+1}_{basename}",
                    "labels": [label],
                }
            )

    print("\nAvailable calibration sets")
    print("-" * 60)
    for s in sets:
        print(f"{s['id']} = {' + '.join(s['labels'])}", flush=True)

    return sets


# ============================================================
# 2x2 PANEL SPECTRA PLOT
# ============================================================

def init_panel_spectra_figure():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    panel_order = ["light-gray", "medium-gray", "dark-gray", "black"]

    for ax, label in zip(axes, panel_order):
        ax.set_title(PANEL_TITLES[label])
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Radiance")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.ion()
    fig.show()

    return fig, dict(zip(panel_order, axes))


def update_panel_spectra_plot(fig, axes, wavelengths, panel_samples, basename):
    for label, spectra_list in panel_samples.items():
        if len(spectra_list) == 0:
            continue

        spectra = [item["spectrum"] for item in spectra_list]
        mean_spectrum = np.mean(np.stack(spectra, axis=0), axis=0)

        ax = axes[label]
        ax.plot(wavelengths, mean_spectrum, label=basename)
        ax.legend(fontsize=8)

    fig.canvas.draw()
    fig.canvas.flush_events()


# ============================================================
# SIMPLE SET-SELECTION GUI
# ============================================================

def choose_sets_gui(all_image_info, all_hsi_basenames):
    import tkinter as tk
    from tkinter import ttk

    global_sets = []

    for source_basename, info in all_image_info.items():
        for s in info["sets"]:
            s_copy = dict(s)
            s_copy["source_basename"] = source_basename
            global_sets.append(s_copy)

    if len(global_sets) == 0:
        raise RuntimeError("No calibration sets were found from any image.")

    print("Opening calibration-set selection GUI...", flush=True)

    root = tk.Tk()
    root.title("Choose calibration sets for ELM")

    selected_vars = {}
    output = {}

    canvas = tk.Canvas(root, width=950, height=650)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    frame = ttk.Frame(canvas)

    frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    row = 0

    ttk.Label(
        frame,
        text="Select calibration sets to use for each image",
        font=("Arial", 14, "bold"),
    ).grid(row=row, column=0, sticky="w", pady=10)

    row += 1

    ttk.Label(
        frame,
        text="Each image can use one or more sets. Sets are shown as: Set ID: panel labels [source image].",
        font=("Arial", 10),
    ).grid(row=row, column=0, sticky="w", pady=(0, 10))

    row += 1

    for basename in all_hsi_basenames:
        ttk.Label(
            frame,
            text=basename,
            font=("Arial", 12, "bold"),
        ).grid(row=row, column=0, sticky="w", pady=(15, 5))

        row += 1

        selected_vars[basename] = {}

        for set_info in global_sets:
            sid = set_info["id"]
            labels = " + ".join(set_info["labels"])
            source = set_info["source_basename"]

            key = f"{source}::{sid}"

            var = tk.BooleanVar(master=root, value=False)
            selected_vars[basename][key] = var

            cb = ttk.Checkbutton(
                frame,
                text=f"{sid}: {labels}    [from {source}]",
                variable=var,
            )
            cb.grid(row=row, column=0, sticky="w", padx=20)

            row += 1

    status_label = ttk.Label(frame, text="", foreground="green")
    status_label.grid(row=row, column=0, sticky="w", pady=(10, 5))
    row += 1

    def submit():
        print("Submit clicked in set-selection GUI.", flush=True)

        for basename, vars_dict in selected_vars.items():
            output[basename] = [
                key for key, var in vars_dict.items() if var.get()
            ]

        print("Selected calibration sets:", flush=True)
        for basename, chosen in output.items():
            print(f"  {basename}: {chosen}", flush=True)

        status_label.config(text="Submitted. Closing window...")

        root.quit()
        root.after(100, root.destroy)

    def on_close():
        print("Set-selection GUI closed without submit.", flush=True)
        root.quit()
        root.after(100, root.destroy)

    submit_btn = ttk.Button(frame, text="Submit", command=submit)
    submit_btn.grid(row=row, column=0, pady=20)

    root.protocol("WM_DELETE_WINDOW", on_close)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    root.mainloop()

    try:
        root.destroy()
    except Exception:
        pass

    print("Returned from set-selection GUI.", flush=True)

    return output


# ============================================================
# ELM FIT + APPLY
# ============================================================

def fit_elm_coefficients_from_selected_global_sets(
    selected_set_keys,
    all_image_info,
    target_wavelengths,
):
    """
    Fit ELM coefficients using selected calibration sets.

    Each extracted panel spectrum carries its own .sig path, selected by the
    user during rectangle drawing.

    This is more accurate than using one global reflectance file per panel
    because the field calibration file is linked directly to the selected ROI.
    """

    X_list = []
    Y_list = []

    reflectance_cache = {}

    for key in selected_set_keys:
        source_basename, set_id = key.split("::", 1)

        if source_basename not in all_image_info:
            print(f"Warning: source image {source_basename} not found.")
            continue

        source_info = all_image_info[source_basename]
        source_panel_samples = source_info["panel_samples"]
        source_wavelengths = source_info["wavelengths"]

        matching_sets = [
            s for s in source_info["sets"]
            if s["id"] == set_id
        ]

        if len(matching_sets) == 0:
            print(f"Warning: set {set_id} not found for {source_basename}.")
            continue

        set_info = matching_sets[0]

        for label in set_info["labels"]:

            if label not in source_panel_samples:
                print(f"Warning: no extracted panel samples for label {label}; skipping.")
                continue

            for sample in source_panel_samples[label]:
                measured_spec = sample["spectrum"]
                sig_path = sample["sig_path"]

                if sig_path not in reflectance_cache:
                    wl, rf = read_sig_reflectance(sig_path)
                    reflectance_cache[sig_path] = (wl, rf)

                ref_wavelengths, ref_reflectance = reflectance_cache[sig_path]

                known_reflectance = np.interp(
                    target_wavelengths,
                    ref_wavelengths,
                    ref_reflectance,
                    left=ref_reflectance[0],
                    right=ref_reflectance[-1],
                ).astype(np.float32)

                if (
                    len(source_wavelengths) != len(target_wavelengths)
                    or not np.allclose(source_wavelengths, target_wavelengths)
                ):
                    measured_spec = np.interp(
                        target_wavelengths,
                        source_wavelengths,
                        measured_spec,
                        left=measured_spec[0],
                        right=measured_spec[-1],
                    ).astype(np.float32)
                else:
                    measured_spec = measured_spec.astype(np.float32)

                X_list.append(measured_spec)
                Y_list.append(known_reflectance)

    if len(X_list) < 2:
        raise RuntimeError(
            "Need at least two calibration panel spectra to fit ELM. "
            "Select more calibration sets."
        )

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)

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

###### SIG FILE SELECTION
def list_sig_files(svc_folder):
    """
    Find all .sig field calibration files in a folder.

    This is important because the user can choose the field calibration
    measurement that corresponds to each selected rectangle while they are
    still looking at the image and field context.
    """
    if svc_folder is None:
        return []

    if not os.path.isdir(svc_folder):
        raise NotADirectoryError(f"SVC folder does not exist: {svc_folder}")

    sig_files = sorted(glob(os.path.join(svc_folder, "*.sig")), key=alphanum_key)

    if len(sig_files) == 0:
        raise RuntimeError(f"No .sig files found in SVC folder: {svc_folder}")

    return sig_files


# ============================================================
# MAIN WORKFLOW
# ============================================================

def run_interactive_elm_calibration(
    hsi_folder,
    output_folder,
    svc_folder,
    rgb_bands=(110, 60, 30),
    nodata_value=0,
    use_existing_coefficients=True,
    collect_panels_for_all_images=False,
    overwrite_reflectance_outputs=True,
):
    """
    If use_existing_coefficients=True:
        - For each cube, check for:
            output_name_rf_elm_slope.npy
            output_name_rf_elm_intercept.npy
        - If both exist and match band count, use them directly.
        - Only run interactive panel selection for cubes missing coefficients.

    If collect_panels_for_all_images=True:
        - Even cubes with existing coefficients are included in the panel collection GUI.
        - Useful if you want their panels to serve as calibration sets for other cubes.
    """
    os.makedirs(output_folder, exist_ok=True)

    hsi_files = list_envi_hsi_files(hsi_folder)
    all_hsi_basenames = [os.path.basename(f) for f in hsi_files]

    if len(hsi_files) == 0:
        raise RuntimeError(f"No ENVI HSI files found in: {hsi_folder}")

    hsi_lookup = {os.path.basename(p): p for p in hsi_files}

    coefficient_status = {}

    print("\nChecking for existing ELM coefficients...")
    print("=" * 80)

    for hsi_path in hsi_files:
        basename = os.path.basename(hsi_path)
        output_path = make_output_path(hsi_path, output_folder, suffix="_rf")

        with rasterio.open(hsi_path) as src:
            n_bands = src.count

        has_coeffs = False

        if use_existing_coefficients and coefficients_exist(output_path):
            try:
                load_coefficients(output_path, expected_n_bands=n_bands)
                has_coeffs = True
                print(f"{basename}: existing coefficients found.", flush=True)
            except Exception as e:
                has_coeffs = False
                print(f"{basename}: coefficient files exist but are invalid: {e}", flush=True)
        else:
            print(f"{basename}: coefficients missing.", flush=True)

        coefficient_status[basename] = {
            "hsi_path": hsi_path,
            "output_path": output_path,
            "has_coefficients": has_coeffs,
        }

    missing_coeff_basenames = [
        b for b, info in coefficient_status.items()
        if not info["has_coefficients"]
    ]

    if use_existing_coefficients and len(missing_coeff_basenames) == 0:
        print("\nAll cubes already have valid coefficients.")
        print("Skipping interactive panel collection and applying existing coefficients.", flush=True)
        all_image_info = {}
        selected_sets = {}
    else:
        if collect_panels_for_all_images:
            basenames_to_collect = all_hsi_basenames
        else:
            basenames_to_collect = missing_coeff_basenames

        print("\nImages requiring panel collection:")
        for b in basenames_to_collect:
            print(f"  {b}", flush=True)

        all_image_info = {}

        # --------------------------------------------------------
        # Step 1: rectangle drawing + spectra extraction
        # --------------------------------------------------------
        for basename in basenames_to_collect:
            hsi_path = hsi_lookup[basename]

            print("\n" + "=" * 80)
            print(f"Loading: {basename}", flush=True)

            with rasterio.open(hsi_path) as src:
                cube = src.read().astype(np.float32)
                profile = src.profile.copy()

            wavelengths = get_cube_wavelengths(hsi_path)

            rgb = make_rgb_preview(
                cube,
                rgb_bands=rgb_bands,
                nodata_value=nodata_value,
            )

            print("Opening rectangle GUI...", flush=True)
            polygons, skipped = collect_polygons_for_cube(
                rgb,
                basename,
                svc_folder=svc_folder,
            )
            print("Returned from rectangle GUI.", flush=True)

            plt.pause(0.2)

            if skipped:
                print(f"Skipped: {basename}", flush=True)
                continue

            print(f"Extracting spectra from {len(polygons)} rectangles...", flush=True)
            panel_samples = extract_polygon_spectra(
                cube,
                polygons,
                nodata_value=nodata_value,
            )
            print("Finished extracting rectangle spectra.", flush=True)

            sets = make_panel_sets(panel_samples, basename)

            all_image_info[basename] = {
                "path": hsi_path,
                "profile": profile,
                "wavelengths": wavelengths,
                "panel_samples": panel_samples,
                "sets": sets,
            }

        missing_after_collection = [
            b for b in missing_coeff_basenames
            if b not in all_image_info
        ]

        if len(all_image_info) == 0 and len(missing_coeff_basenames) > 0:
            raise RuntimeError(
                "Some cubes are missing coefficients, but no calibration panels were submitted."
            )

        # --------------------------------------------------------
        # Step 1b: spectra summary plot after all collection
        # --------------------------------------------------------
        if len(all_image_info) > 0:
            print("Finished collecting panel spectra.", flush=True)
            print("Opening 2x2 panel spectra summary figure...", flush=True)

            fig, axes = init_panel_spectra_figure()

            for basename, info in all_image_info.items():
                update_panel_spectra_plot(
                    fig,
                    axes,
                    info["wavelengths"],
                    info["panel_samples"],
                    basename,
                )

            plt.show(block=False)
            plt.pause(0.5)

            # --------------------------------------------------------
            # Step 2: user selects calibration sets
            # --------------------------------------------------------
            print("About to open set-selection GUI.", flush=True)

            if collect_panels_for_all_images:
                gui_basenames = all_hsi_basenames
            else:
                gui_basenames = missing_coeff_basenames

            selected_sets = choose_sets_gui(all_image_info, gui_basenames)

            print("Set-selection GUI finished.", flush=True)
            print(selected_sets, flush=True)
        else:
            selected_sets = {}

    # --------------------------------------------------------
    # Step 3: fit/load ELM and apply per image
    # --------------------------------------------------------
    for basename in all_hsi_basenames:
        print("\n" + "=" * 80)
        print(f"Calibrating: {basename}", flush=True)

        hsi_path = hsi_lookup[basename]
        output_path = make_output_path(hsi_path, output_folder, suffix="_rf")

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

        if use_existing_coefficients and coefficient_status[basename]["has_coefficients"]:
            print("Using existing ELM coefficients.", flush=True)
            slopes, intercepts = load_coefficients(
                output_path,
                expected_n_bands=n_bands,
            )
        else:
            chosen = selected_sets.get(basename, [])

            if len(chosen) == 0:
                print(f"No sets selected for {basename}; skipping calibration.", flush=True)
                continue

            slopes, intercepts = fit_elm_coefficients_from_selected_global_sets(
                selected_set_keys=chosen,
                all_image_info=all_image_info,
                target_wavelengths=wavelengths,
            )

            save_coefficients(output_path, slopes, intercepts)

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

    run_interactive_elm_calibration(
        hsi_folder="/mnt/data-drive/projects_data/darpa/beets/2022/4th_20220810/nano/1430_radiance_coreg",
        output_folder="/mnt/data-drive/projects_data/darpa/beets/2022/4th_20220810/nano/1430_radiance_coreg_ref",

        # Folder containing all field calibration .sig files.
        # The user will choose the correct .sig file inside the same
        # rectangle-picking GUI before drawing each panel ROI.
        svc_folder="/mnt/data-drive/projects_data/darpa/beets/2022/4th_20220810/SVC",

        rgb_bands=(110, 60, 30),
        nodata_value=0,

        use_existing_coefficients=True,
        collect_panels_for_all_images=False,
        overwrite_reflectance_outputs=True,
    )