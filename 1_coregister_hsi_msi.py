import os
import cv2
import numpy as np
import rasterio

from rasterio.transform import xy
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling, transform_bounds


# -----------------------------
# Basic image utilities
# -----------------------------
def normalize_for_registration(img, nodata_value=0):
    img = img.astype(np.float32)

    valid = np.isfinite(img)
    if nodata_value is not None:
        valid &= img != nodata_value

    out = np.zeros_like(img, dtype=np.float32)

    if valid.sum() == 0:
        return out, valid.astype(np.uint8)

    lo, hi = np.nanpercentile(img[valid], [2, 98])
    out[valid] = np.clip((img[valid] - lo) / (hi - lo + 1e-8), 0, 1)

    return out.astype(np.float32), valid.astype(np.uint8)


def to_uint8_for_features(img):
    img = np.nan_to_num(img, nan=0.0)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def make_rgb_composite(cube, band_indices, nodata_value=0):
    """
    cube: [bands, rows, cols]
    band_indices: list of 3 zero-based band indices

    returns:
        rgb: [rows, cols, 3], float32 normalized 0-1
        valid: [rows, cols], uint8
    """
    if band_indices is None or len(band_indices) != 3:
        raise ValueError("band_indices must be a list of exactly 3 zero-based band indices.")

    rgb_raw = cube[band_indices].transpose(1, 2, 0).astype(np.float32)

    valid = np.all(np.isfinite(rgb_raw), axis=-1)
    if nodata_value is not None:
        valid &= np.all(rgb_raw != nodata_value, axis=-1)

    rgb = np.zeros_like(rgb_raw, dtype=np.float32)

    for i in range(3):
        band = rgb_raw[:, :, i]
        band_valid = valid & np.isfinite(band)

        if band_valid.sum() > 0:
            lo, hi = np.nanpercentile(band[band_valid], [2, 98])
            rgb[:, :, i][band_valid] = np.clip(
                (band[band_valid] - lo) / (hi - lo + 1e-8),
                0,
                1
            )

    rgb[~valid] = 0
    return rgb.astype(np.float32), valid.astype(np.uint8)


def rgb_to_gray(rgb):
    return cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2GRAY)


# -----------------------------
# Raster/geospatial utilities
# -----------------------------
def get_hsi_bounds_in_msi_crs(hsi_path, msi_reference_path):
    """
    Return HSI bounds transformed into the MicaSense/MSI CRS.
    """
    with rasterio.open(hsi_path) as hsi:
        hsi_bounds = hsi.bounds
        hsi_crs = hsi.crs

    with rasterio.open(msi_reference_path) as msi:
        msi_crs = msi.crs

    if hsi_crs is None:
        raise ValueError("HSI file has no CRS.")
    if msi_crs is None:
        raise ValueError("MSI/MicaSense file has no CRS.")

    if hsi_crs != msi_crs:
        hsi_bounds = transform_bounds(
            hsi_crs,
            msi_crs,
            *hsi_bounds,
            densify_pts=21
        )

    return hsi_bounds


def read_msi_roi_in_msi_crs(msi_reference_path, bounds_in_msi_crs):
    """
    Crop the MicaSense/MSI mosaic using bounds already expressed in MSI CRS.
    """
    with rasterio.open(msi_reference_path) as ref:
        window = from_bounds(*bounds_in_msi_crs, transform=ref.transform)
        window = window.round_offsets().round_lengths()

        ref_data = ref.read(window=window)
        ref_transform = ref.window_transform(window)
        ref_profile = ref.profile.copy()
        ref_crs = ref.crs

    return ref_data, ref_transform, ref_profile, ref_crs


def reproject_hsi_to_msi_roi_grid(
    hsi_path,
    ref_transform,
    ref_crs,
    out_height,
    out_width,
    resampling=Resampling.nearest,
):
    """
    Reproject full HSI cube to the cropped MicaSense/MSI ROI grid.
    Output is in MicaSense CRS and pixel grid.
    """
    with rasterio.open(hsi_path) as src:
        hsi_reproj = np.zeros(
            (src.count, out_height, out_width),
            dtype=np.float32
        )

        for b in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, b),
                destination=hsi_reproj[b - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=resampling,
                dst_nodata=0,
            )

    return hsi_reproj

def read_hsi_cube_and_grid(hsi_path):
    """
    Read the HSI cube and keep its native CRS, transform, shape, and profile.

    This preserves the native HSI spatial resolution.
    """
    with rasterio.open(hsi_path) as src:
        hsi_cube = src.read().astype(np.float32)
        hsi_transform = src.transform
        hsi_crs = src.crs
        hsi_profile = src.profile.copy()
        hsi_height = src.height
        hsi_width = src.width

    if hsi_crs is None:
        raise ValueError("HSI file has no CRS.")

    return hsi_cube, hsi_transform, hsi_crs, hsi_profile, hsi_height, hsi_width


def read_hsi_cube_and_grid(hsi_path):
    """
    Read the HSI cube and keep its native CRS, transform, shape, and profile.

    This preserves the native HSI spatial resolution.
    """
    with rasterio.open(hsi_path) as src:
        hsi_cube = src.read().astype(np.float32)
        hsi_transform = src.transform
        hsi_crs = src.crs
        hsi_profile = src.profile.copy()
        hsi_height = src.height
        hsi_width = src.width

    if hsi_crs is None:
        raise ValueError("HSI file has no CRS.")

    return hsi_cube, hsi_transform, hsi_crs, hsi_profile, hsi_height, hsi_width


def reproject_msi_to_hsi_grid(
    msi_reference_path,
    hsi_transform,
    hsi_crs,
    out_height,
    out_width,
    resampling=Resampling.nearest,
    dst_nodata=0,
):
    """
    Reproject the MicaSense/MSI mosaic to the native HSI grid.

    This is the key change:
    instead of lowering or changing the HSI resolution to match MSI,
    the MSI is resampled into the HSI CRS, transform, height, and width.

    Output:
        msi_on_hsi_grid: [bands, HSI_rows, HSI_cols]
    """
    with rasterio.open(msi_reference_path) as src:
        if src.crs is None:
            raise ValueError("MSI/MicaSense file has no CRS.")

        msi_on_hsi_grid = np.zeros(
            (src.count, out_height, out_width),
            dtype=np.float32
        )

        for b in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, b),
                destination=msi_on_hsi_grid[b - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=hsi_transform,
                dst_crs=hsi_crs,
                resampling=resampling,
                dst_nodata=dst_nodata,
            )

    return msi_on_hsi_grid

def pixel_to_map_coords(points_xy, transform):
    """
    Convert pixel coordinates to map coordinates.

    points_xy: [N, 2], x=col, y=row
    transform: rasterio affine transform
    """
    cols = points_xy[:, 0]
    rows = points_xy[:, 1]

    xs, ys = xy(transform, rows, cols, offset="center")
    return np.column_stack([xs, ys]).astype(np.float32)


# -----------------------------
# SIFT + geolocation homography
# -----------------------------
def estimate_sift_homography_with_geolocation(
    fixed,
    moving,
    ref_transform,
    valid_mask=None,
    nfeatures=5000,
    ratio_thresh=0.75,
    ransac_thresh=5.0,
    max_geo_distance=None,
):
    """
    fixed:
        MSI/MicaSense reference image, [H, W], normalized float32.

    moving:
        HSI image already reprojected to MSI grid, [H, W], normalized float32.

    ref_transform:
        Rasterio transform for the MSI ROI grid.

    Returns:
        H: 3x3 homography mapping moving/HSI pixels -> fixed/MSI pixels.
        info: diagnostics.
    """

    fixed_u8 = to_uint8_for_features(fixed)
    moving_u8 = to_uint8_for_features(moving)

    if valid_mask is not None:
        mask_u8 = (valid_mask > 0).astype(np.uint8) * 255
    else:
        mask_u8 = None

    sift = cv2.SIFT_create(nfeatures=nfeatures)

    kp_fixed, des_fixed = sift.detectAndCompute(fixed_u8, mask_u8)
    kp_moving, des_moving = sift.detectAndCompute(moving_u8, mask_u8)

    if des_fixed is None or des_moving is None:
        raise RuntimeError("SIFT could not find enough descriptors.")

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = matcher.knnMatch(des_moving, des_fixed, k=2)

    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        raise RuntimeError(f"Not enough good SIFT matches after ratio test: {len(good_matches)}")

    src_pts = np.float32(
        [kp_moving[m.queryIdx].pt for m in good_matches]
    )  # HSI/moving pixels

    dst_pts = np.float32(
        [kp_fixed[m.trainIdx].pt for m in good_matches]
    )  # MSI/fixed pixels

    # Geolocation filtering
    if max_geo_distance is not None:
        src_geo = pixel_to_map_coords(src_pts, ref_transform)
        dst_geo = pixel_to_map_coords(dst_pts, ref_transform)

        geo_dist = np.linalg.norm(src_geo - dst_geo, axis=1)
        keep = geo_dist <= max_geo_distance

        src_pts = src_pts[keep]
        dst_pts = dst_pts[keep]
        good_matches = [m for m, k in zip(good_matches, keep) if k]

        if len(good_matches) < 4:
            raise RuntimeError(
                f"Not enough matches after geolocation filtering: {len(good_matches)}"
            )

        print(f"Matches after geolocation filter: {len(good_matches)}")
        print(f"Median geolocation distance before filter: {np.median(geo_dist):.3f}")
        print(f"Max accepted geolocation distance: {max_geo_distance:.3f}")

    src_pts_cv = src_pts.reshape(-1, 1, 2)
    dst_pts_cv = dst_pts.reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(
        src_pts_cv,
        dst_pts_cv,
        cv2.RANSAC,
        ransac_thresh
    )

    if H is None:
        raise RuntimeError("Homography estimation failed.")

    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else len(good_matches)

    print(f"SIFT keypoints fixed/MSI:  {len(kp_fixed)}")
    print(f"SIFT keypoints moving/HSI: {len(kp_moving)}")
    print(f"Good matches used:         {len(good_matches)}")
    print(f"RANSAC inliers:            {n_inliers}")

    info = {
        "kp_fixed": kp_fixed,
        "kp_moving": kp_moving,
        "matches": good_matches,
        "inlier_mask": inlier_mask,
        "src_pts": src_pts,
        "dst_pts": dst_pts,
        "n_good_matches": len(good_matches),
        "n_inliers": n_inliers,
    }

    return H.astype(np.float32), info


def warp_hsi_cube_homography(
    hsi_cube,
    H,
    interpolation=cv2.INTER_NEAREST,
    nodata_value=0,
):
    """
    Apply homography to all HSI bands.

    H maps HSI/moving pixel coordinates -> MSI/fixed pixel coordinates.
    """
    bands, rows, cols = hsi_cube.shape
    out = np.zeros_like(hsi_cube, dtype=np.float32)

    for b in range(bands):
        out[b] = cv2.warpPerspective(
            hsi_cube[b].astype(np.float32),
            H,
            (cols, rows),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=nodata_value,
        )

    return out


# -----------------------------
# Save ENVI
# -----------------------------
def read_envi_wavelengths(hsi_path):
    hdr_path = hsi_path + ".hdr"

    if not os.path.exists(hdr_path):
        return None

    with open(hdr_path, "r", errors="ignore") as f:
        text = f.read()

    import re
    match = re.search(r"wavelength\s*=\s*\{([^}]*)\}", text, flags=re.IGNORECASE | re.DOTALL)

    if match is None:
        return None

    values = match.group(1).replace("\n", " ").split(",")

    wavelengths = []
    for v in values:
        v = v.strip()
        if v:
            wavelengths.append(float(v))

    return np.array(wavelengths, dtype=np.float32)


def append_envi_wavelengths(output_path, wavelengths):
    if wavelengths is None:
        return

    hdr_path = output_path + ".hdr"

    with open(hdr_path, "a") as f:
        f.write("\n")
        f.write("wavelength units = Nanometers\n")
        f.write("wavelength = {\n")

        vals = [f"{float(w):.6f}" for w in wavelengths]

        for i in range(0, len(vals), 5):
            line = ", ".join(vals[i:i+5])
            if i + 5 < len(vals):
                line += ","
            f.write("  " + line + "\n")

        f.write("}\n")

def save_envi(output_path, cube, ref_profile, ref_transform, ref_crs, wavelengths=None):
    out_profile = ref_profile.copy()

    out_profile.update(
        {
            "driver": "ENVI",
            "height": cube.shape[1],
            "width": cube.shape[2],
            "count": cube.shape[0],
            "dtype": "float32",
            "transform": ref_transform,
            "crs": ref_crs,
            "interleave": "bsq",
        }
    )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(cube.astype(np.float32))

    append_envi_wavelengths(output_path, wavelengths)

    print(f"Saved coregistered HSI cube to: {output_path}")


# -----------------------------
# Main workflow
# -----------------------------
def coregister_hsi_to_micasense_homography(
    hsi_path,
    msi_reference_path,
    output_path,
    hsi_reg_bands,
    msi_reg_bands,
    hsi_nodata=0,
    msi_nodata=None,
    nfeatures=5000,
    ratio_thresh=0.75,
    ransac_thresh=5.0,
    max_geo_distance=None,
):
    """
    Coregister one HSI cube to one MicaSense/MSI mosaic using SIFT,
    geolocation filtering, and homography.

    Important:
    The MSI/MicaSense mosaic is reprojected to the native HSI grid.
    This preserves the HSI spatial resolution instead of forcing the HSI
    onto the MSI pixel grid.
    """

    # 1. Read HSI in its native grid
    hsi_cube, hsi_transform, hsi_crs, hsi_profile, hsi_height, hsi_width = (
        read_hsi_cube_and_grid(hsi_path)
    )

    # 2. Reproject MSI/MicaSense to the native HSI grid
    #
    # This is important because the registration should not automatically
    # lower, raise, or otherwise change the HSI spatial grid before matching.
    # By placing MSI on the HSI grid, the final output remains in the HSI
    # native CRS, transform, pixel size, height, and width.
    effective_msi_nodata = 0 if msi_nodata is None else msi_nodata

    msi_on_hsi_grid = reproject_msi_to_hsi_grid(
        msi_reference_path=msi_reference_path,
        hsi_transform=hsi_transform,
        hsi_crs=hsi_crs,
        out_height=hsi_height,
        out_width=hsi_width,
        resampling=Resampling.nearest,
        dst_nodata=effective_msi_nodata,
    )

    # 3. Build 3-band composites on the same HSI grid
    hsi_rgb, hsi_valid = make_rgb_composite(
        hsi_cube,
        hsi_reg_bands,
        nodata_value=hsi_nodata
    )

    msi_rgb, msi_valid = make_rgb_composite(
        msi_on_hsi_grid,
        msi_reg_bands,
        nodata_value=effective_msi_nodata
    )

    fixed = rgb_to_gray(msi_rgb)
    moving = rgb_to_gray(hsi_rgb)

    valid_mask = (hsi_valid & msi_valid).astype(np.uint8)

    # 4. Set geolocation filter if not provided
    #
    # max_geo_distance is now based on HSI pixel size because both images
    # are being matched in the HSI grid.
    if max_geo_distance is None:
        pixel_size_x = abs(hsi_transform.a)
        pixel_size_y = abs(hsi_transform.e)
        pixel_size = max(pixel_size_x, pixel_size_y)

        # This is important:
        # max_geo_distance limits how far a SIFT match is allowed to move
        # in real map units before homography estimation.
        # It prevents repeated vegetation rows, panels, or similar texture
        # patterns from creating geospatially unrealistic matches.
        max_geo_distance = 70 * pixel_size

    # 5. Estimate homography
    #
    # H maps HSI/moving pixel coordinates -> MSI/fixed pixel coordinates.
    # Since MSI has been reprojected to the HSI grid, this homography is
    # estimated directly in HSI-resolution pixel space.
    H, match_info = estimate_sift_homography_with_geolocation(
        fixed=fixed,
        moving=moving,
        ref_transform=hsi_transform,
        valid_mask=valid_mask,
        nfeatures=nfeatures,
        ratio_thresh=ratio_thresh,
        ransac_thresh=ransac_thresh,
        max_geo_distance=max_geo_distance,
    )

    # 6. Apply homography to every HSI band
    hsi_coreg = warp_hsi_cube_homography(
        hsi_cube=hsi_cube,
        H=H,
        interpolation=cv2.INTER_NEAREST,
        nodata_value=hsi_nodata,
    )

    wavelengths = read_envi_wavelengths(hsi_path)

    if wavelengths is not None and len(wavelengths) != hsi_coreg.shape[0]:
        print(
            f"Warning: wavelength count {len(wavelengths)} does not match "
            f"cube band count {hsi_coreg.shape[0]}. Not writing wavelengths."
        )
        wavelengths = None

    # 7. Save output in native HSI CRS/grid
    save_envi(
        output_path=output_path,
        cube=hsi_coreg,
        ref_profile=hsi_profile,
        ref_transform=hsi_transform,
        ref_crs=hsi_crs,
        wavelengths=wavelengths,
    )

    return H, match_info, hsi_coreg

def bounds_intersect(bounds_a, bounds_b):
    """
    Check whether two bounding boxes intersect.

    bounds_a, bounds_b:
        rasterio BoundingBox or tuple/list:
        (left, bottom, right, top)
    """
    a_left, a_bottom, a_right, a_top = bounds_a
    b_left, b_bottom, b_right, b_top = bounds_b

    return not (
        a_right <= b_left or
        a_left >= b_right or
        a_top <= b_bottom or
        a_bottom >= b_top
    )


def get_msi_bounds(msi_reference_path):
    with rasterio.open(msi_reference_path) as src:
        return src.bounds, src.crs


def hsi_intersects_msi(hsi_path, msi_reference_path):
    """
    Returns True if HSI footprint intersects the MSI/MicaSense mosaic.

    HSI bounds are transformed into the MSI CRS before intersection testing.
    """
    hsi_bounds_msi_crs = get_hsi_bounds_in_msi_crs(
        hsi_path,
        msi_reference_path
    )

    msi_bounds, _ = get_msi_bounds(msi_reference_path)

    return bounds_intersect(hsi_bounds_msi_crs, msi_bounds)


def make_output_path_for_hsi(hsi_path, output_folder, suffix="_wrp", extension=".dat"):
    """
    Create output filename in output_folder with suffix added.

    Example:
        raw_9631_rd_or -> output_folder/raw_9631_rd_or_wrp.dat
        cube.tif       -> output_folder/cube_wrp.dat
    """
    basename = os.path.basename(hsi_path)

    return os.path.join(output_folder, f"{basename}{suffix}")


def list_hsi_files(hsi_folder, patterns=None):
    """
    Find HSI data files in a folder.

    This skips ENVI .hdr files and keeps:
    - regular raster files with known extensions
    - extensionless ENVI data files that have a matching .hdr
    """
    import os
    from glob import glob

    if patterns is None:
        patterns = [
            "*.tif",
            "*.tiff",
            "*.dat",
            "*.img",
            "*.bil",
            "*.bsq",
            "*.bip",
            "*",   # needed for extensionless ENVI data files
        ]

    files = []
    for pattern in patterns:
        files.extend(glob(os.path.join(hsi_folder, pattern)))

    clean_files = []
    for f in files:
        if os.path.isdir(f):
            continue

        if f.lower().endswith(".hdr"):
            continue

        base, ext = os.path.splitext(f)

        # Keep normal raster files
        if ext.lower() in [".tif", ".tiff", ".dat", ".img", ".bil", ".bsq", ".bip"]:
            clean_files.append(f)
            continue

        # Keep extensionless ENVI data files only if matching .hdr exists
        if ext == "" and os.path.exists(f + ".hdr"):
            clean_files.append(f)

    return sorted(set(clean_files))


def coregister_hsi_folder_to_micasense(
    hsi_folder,
    msi_reference_path,
    output_folder,
    hsi_reg_bands,
    msi_reg_bands,
    hsi_nodata=0,
    msi_nodata=None,
    nfeatures=8000,
    ratio_thresh=0.75,
    ransac_thresh=5.0,
    max_geo_distance=None,
    patterns=None,
    skip_existing=True,
):
    """
    Coregister all HSI files in a folder to one MicaSense/MSI reference mosaic.

    - HSI files outside MSI bounds are skipped.
    - Outputs are written to output_folder.
    - Output filenames get '_wrp.dat' appended.
    """

    os.makedirs(output_folder, exist_ok=True)

    hsi_files = list_hsi_files(hsi_folder, patterns=patterns)

    if len(hsi_files) == 0:
        raise RuntimeError(f"No HSI files found in folder: {hsi_folder}")

    print(f"Found {len(hsi_files)} candidate HSI files.")
    print(f"Output folder: {output_folder}")

    results = []

    for idx, hsi_path in enumerate(hsi_files, start=1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(hsi_files)}] Processing:")
        print(hsi_path)

        output_path = make_output_path_for_hsi(
            hsi_path=hsi_path,
            output_folder=output_folder,
            suffix="_wrp",
            extension="",
        )

        if skip_existing and os.path.exists(output_path):
            print(f"Skipping because output already exists: {output_path}")
            results.append(
                {
                    "hsi_path": hsi_path,
                    "output_path": output_path,
                    "status": "skipped_existing",
                    "error": None,
                }
            )
            continue

        try:
            # Make sure the HSI overlaps with the MSI mosaic before processing
            if not hsi_intersects_msi(hsi_path, msi_reference_path):
                print("Skipping: HSI footprint is outside MicaSense/MSI bounds.")
                results.append(
                    {
                        "hsi_path": hsi_path,
                        "output_path": output_path,
                        "status": "skipped_outside_bounds",
                        "error": None,
                    }
                )
                continue

            H, match_info, _ = coregister_hsi_to_micasense_homography(
                hsi_path=hsi_path,
                msi_reference_path=msi_reference_path,
                output_path=output_path,
                hsi_reg_bands=hsi_reg_bands,
                msi_reg_bands=msi_reg_bands,
                hsi_nodata=hsi_nodata,
                msi_nodata=msi_nodata,
                nfeatures=nfeatures,
                ratio_thresh=ratio_thresh,
                ransac_thresh=ransac_thresh,
                max_geo_distance=max_geo_distance,
            )

            # Save homography next to output
            homography_path = output_path + "_H.npy"
            np.save(homography_path, H)

            results.append(
                {
                    "hsi_path": hsi_path,
                    "output_path": output_path,
                    "homography_path": homography_path,
                    "status": "processed",
                    "n_good_matches": match_info.get("n_good_matches"),
                    "n_inliers": match_info.get("n_inliers"),
                    "error": None,
                }
            )

        except Exception as e:
            print(f"Failed: {e}")
            results.append(
                {
                    "hsi_path": hsi_path,
                    "output_path": output_path,
                    "status": "failed",
                    "error": str(e),
                }
            )

    return results

if __name__ == "__main__":
    results = coregister_hsi_folder_to_micasense(
        hsi_folder="/mnt/data-drive/projects_data/darpa/beets/2022/4th_20220810/nano/1430_radiance",
        msi_reference_path="/mnt/data-drive/projects_data/darpa/beets/2022/4th_20220810/micasense/Geneva-Beets-08-10-2022-orthophoto.tif",
        output_folder="/mnt/data-drive/projects_data/darpa/beets/2022/4th_20220810/nano/1430_radiance_coreg",

        hsi_reg_bands=[114, 69, 24],
        msi_reg_bands=[0, 1, 2],

        hsi_nodata=0,
        msi_nodata=None,

        nfeatures=8000,
        ratio_thresh=0.75,
        ransac_thresh=5.0,

        # None means it will use 5 * MSI pixel size
        #### I open an hsi and the msi ortho and see for a single feature how far off they are from one another
        #### and use that as distance + some
        max_geo_distance=4,

        # Adjust if your files do not have extensions
        patterns=["*"],

        skip_existing=True,
    )

    print("\nSummary")
    print("=" * 80)
    for r in results:
        print(r)