import os
from pathlib import Path
import os

# Force GDAL/PROJ paths
os.environ["PROJ_LIB"] = "/home/axhcis/anaconda3/envs/geo-harmony311/share/proj"
os.environ["PROJ_DATA"] = "/home/axhcis/anaconda3/envs/geo-harmony311/share/proj"
os.environ["GDAL_DATA"] = "/home/axhcis/anaconda3/envs/geo-harmony311/share/gdal"

# Optional but useful for debugging
print("PROJ_LIB =", os.environ.get("PROJ_LIB"))
print("PROJ_DATA =", os.environ.get("PROJ_DATA"))
print("GDAL_DATA =", os.environ.get("GDAL_DATA"))

from osgeo import gdal
gdal.UseExceptions()

def find_envi_binaries(input_dir):
    input_dir = Path(input_dir)
    rasters = []

    for hdr in input_dir.rglob("*.hdr"):
        base = hdr.with_suffix("")

        # Prefer exact basename without extension first
        candidates = [
            base,
            base.with_suffix(".img"),
            base.with_suffix(".dat"),
            base.with_suffix(".bin"),
            base.with_suffix(".raw"),
        ]

        match = None
        for c in candidates:
            if c.exists() and c.is_file():
                match = c
                break

        if match is None:
            print(f"Skipping {hdr}: no matching binary found")
            continue

        ds = gdal.Open(str(match))
        if ds is None:
            print(f"Skipping {match}: GDAL cannot open it")
            continue

        rasters.append(str(match))
        ds = None

    return sorted(rasters)


def inspect_rasters(rasters):
    print("\nInput rasters:")
    for r in rasters:
        ds = gdal.Open(r)
        print(
            os.path.basename(r),
            "| size:", ds.RasterXSize, ds.RasterYSize,
            "| bands:", ds.RasterCount,
            "| proj exists:", bool(ds.GetProjection()),
            "| geotransform:", ds.GetGeoTransform()
        )
        ds = None


def copy_envi_metadata(src_file, out_file):
    src = gdal.Open(src_file)
    out = gdal.Open(out_file, gdal.GA_Update)

    if src is None or out is None:
        raise RuntimeError("Could not open source or output for metadata copy.")

    envi_md = src.GetMetadata("ENVI")

    if envi_md:
        out.SetMetadata(envi_md, "ENVI")

        # These are the important hyperspectral fields.
        for key in [
            "wavelength",
            "fwhm",
            "wavelength units",
            "band names",
            "default bands",
            "data ignore value",
            "sensor type",
        ]:
            if key in envi_md:
                out.SetMetadataItem(key, envi_md[key], "ENVI")

    # Also copy band descriptions if available
    for b in range(1, min(src.RasterCount, out.RasterCount) + 1):
        desc = src.GetRasterBand(b).GetDescription()
        if desc:
            out.GetRasterBand(b).SetDescription(desc)

    out.FlushCache()
    out = None
    src = None


def mosaic_envi(input_dir, output_base):
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    vrt_file = str(output_base.with_suffix(".vrt"))
    out_img = str(output_base)

    rasters = find_envi_binaries(input_dir)

    if len(rasters) == 0:
        raise RuntimeError("No valid ENVI rasters found.")

    print(f"Found {len(rasters)} rasters")
    inspect_rasters(rasters)

    print("\nBuilding VRT...")

    vrt_options = gdal.BuildVRTOptions(
        resolution="highest",
        srcNodata=0,
        VRTNodata=0,
        separate=False,
    )

    vrt = gdal.BuildVRT(
        vrt_file,
        rasters,
        options=vrt_options
    )

    if vrt is None:
        raise RuntimeError("gdal.BuildVRT failed.")

    vrt.FlushCache()
    vrt = None

    print("Writing ENVI mosaic...")

    translate_options = gdal.TranslateOptions(
        format="ENVI",
        creationOptions=[
            "INTERLEAVE=BSQ"
        ]
    )

    gdal.Translate(
        out_img,
        vrt_file,
        options=translate_options
    )

    print("Copying ENVI wavelength metadata...")
    copy_envi_metadata(rasters[0], out_img)

    print("\nDone")
    print("VRT:", vrt_file)
    print("IMG:", out_img)
    print("HDR:", str(output_base.with_suffix(".hdr")))


# Example
mosaic_envi(
    input_dir="/mnt/data-drive/projects_data/darpa/beets/2022/4th_20220810/nano/1430_radiance_coreg_ref_global",
    output_base="/mnt/data-drive/projects_data/darpa/beets/2022/4th_20220810/nano/4th_20220810_mosaic_ref_global"
)