import numpy as np
from pyproj import Transformer

try:
    import contourpy as cp
except ModuleNotFoundError:
    cp = None


# =============================================================================
# Isobath / contour helpers
# =============================================================================

def _contour_arclen(v):
    """
    Arc-length of a contour vertex list v[:, (x,y)] in the same units as v (typically meters).
    """
    dv = np.diff(v, axis=0)
    return np.sum(np.sqrt((dv**2).sum(axis=1)))


def longest_boundary_contour(ds, boundary_mask, crs="EPSG:3031"):
    """
    Extract the longest boundary contour of a mask.

    What this does
    --------------
    - Treats `boundary_mask` as a binary field (0/1) and extracts its boundary by contouring at
      the midpoint value (0.5), which corresponds to the 0/1 interface.
    - Among all contour segments, selects the longest segment (by arc-length in a projected CRS).
    - Returns both:
        (i) the contour vertex list in projected meters (for geometry),
        (ii) the same contour in lon/lat (for plotting).

    Requirements
    ------------
    - `ds` must contain ds["lon"], ds["lat"] as 2D arrays on the same grid as boundary_mask.

    Parameters
    ----------
    ds : xarray.Dataset
        Must contain ds["lon"], ds["lat"] (2D).
    boundary_mask : xarray.DataArray
        Binary/boolean mask on the same horizontal grid as ds["lon"]/ds["lat"].
    crs : str
        Projected CRS used internally for length/geometry (meters).

    Returns
    -------
    contour_xy : (N,2) ndarray
        Contour vertices in projected coordinates (meters).
    contour_lon, contour_lat : (N,) ndarrays
        Same contour vertices in degrees (EPSG:4326).
    """
    from .geometry import _project_lonlat

    lon2d = ds["lon"]
    lat2d = ds["lat"]
    x2d, y2d = _project_lonlat(lon2d, lat2d, crs_out=crs)

    m = boundary_mask.astype(float).fillna(0.0)

    if cp is None:
        raise ModuleNotFoundError(
            "contourpy is required to extract boundary contours. Install contourpy."
        )
    cg = cp.contour_generator(x=x2d, y=y2d, z=np.asarray(m))
    segs = cg.lines(0.5)

    if len(segs) == 0:
        raise ValueError("No contour found for boundary_mask (level=0.5).")

    lens = np.array([_contour_arclen(v) for v in segs])
    contour_xy = segs[int(np.argmax(lens))]

    tf_inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    contour_lon, contour_lat = tf_inv.transform(contour_xy[:, 0], contour_xy[:, 1])
    return contour_xy, np.asarray(contour_lon), np.asarray(contour_lat)


def longest_contour(ds, boundary_mask, crs="EPSG:3031"):
    """
    Backwards-compatible alias for longest_boundary_contour.
    """
    return longest_boundary_contour(ds, boundary_mask, crs=crs)
