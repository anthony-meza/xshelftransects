import numpy as np
import xarray as xr
from pyproj import Transformer

from .geometry import (
    _build_transect_geometry_dataset,
    _contour_outward_sign,
    _make_transect_lonlat,
    _resample_contour_xy,
    _tangent_normal_xy,
)
from .isobath import longest_boundary_contour
from .sampling import (
    _sample_vars,
    _sample_vars_xarray,
    _sample_vars_xesmf,
    _unstack_loc,
)


# =============================================================================
# Public API
# =============================================================================

def xesmf_interpolation(ds_in, obj, lon_t, lat_t, method="bilinear", regridder=None):
    """
    Public wrapper around the xESMF LocStream sampling helper.
    """
    return _sample_vars_xesmf(
        ds_in,
        obj,
        lon_t,
        lat_t,
        method,
        regridder=regridder,
    )


def xarray_interpolation(obj, lon_t, lat_t, method="bilinear", lon_name="lon", lat_name="lat"):
    """
    Public wrapper around the xarray.interp sampling helper.
    """
    return _sample_vars_xarray(obj, lon_t, lat_t, method, lon_name=lon_name, lat_name=lat_name)


def cross_shelf_transects(
    ds,
    var,                    # str | list[str]
    boundary_mask,          # DataArray; binary mask where the 0/1 interface defines transect_length=0
    transect_length=np.arange(0.0, 200e3 + 2e3, 2e3),
    transect_spacing=10e3,
    crs="EPSG:3031",
    engine="xesmf",
    method="bilinear",
    lon_name="lon",
    lat_name="lat",
    return_geometry=True,
):
    """
    Construct and sample cross-shelf transects from a mask boundary.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 2D lon/lat coordinates and bathymetry.
    var : str or list[str]
        Variable name(s) in ds to sample along transects.
    boundary_mask : xarray.DataArray
        Binary mask whose 0/1 interface defines the transect anchor line
        (transect_length = 0). The interface is extracted by contouring
        at the midpoint value (0.5).
    transect_length : 1D array-like
        Cross-shelf distances (meters). Convention is transect_length >= 0 (offshore direction).
    transect_spacing : float
        Target spacing along the boundary between successive sections (meters).
    crs : str
        Projected CRS used for along-boundary geometry (meters), e.g., "EPSG:3031"
        for Antarctica, "EPSG:3413" for the Arctic, or "EPSG:3857" for a more
        general global projection.
    engine : {"xesmf", "xarray"}
        Sampling backend. "xesmf" supports curvilinear grids; "xarray" requires 1D lon/lat.
    method : str
        Interpolation method for sampling (e.g., "bilinear", "nearest_s2d").
    lon_name, lat_name : str
        Coordinate names for lon/lat when engine="xarray".
    return_geometry : bool
        If True, return the geometry dataset in addition to sampled values.

    Returns
    -------
    xshelf : xarray.DataArray or xarray.Dataset
        Sampled values with dims (..., section, transect_length). If `var` is a string, returns a DataArray;
        if `var` is a list, returns a Dataset.
    geometry : xarray.Dataset
        Returned only if return_geometry=True. Contains:
          - anchor lon0/lat0 (section)
          - along-boundary distance s_m (section)
          - normals nx/ny (section)
          - sampled depth along transects depth_xshelf (section, transect_length)
          - boundary contour lon/lat (contour_pt)

    Notes
    -----
    The boundary is extracted as the longest contour of `boundary_mask` at the 0/1
    midpoint (0.5), resampled at `transect_spacing`. Transects are oriented by
    contour winding so +transect_length points away from the enclosed mask, and
    land/ice (deptho <= 0) are masked to NaN.
    """
    if ("lon" not in ds) or ("lat" not in ds):
        raise KeyError(
            "cross_shelf_transects requires ds['lon'] and ds['lat'] as 2D horizontal coordinates. "
            "Rename your coordinate variables to 'lon'/'lat'."
        )
    if "deptho" not in ds:
        raise KeyError(
            "cross_shelf_transects requires ds['deptho'] (ocean depth in meters, positive downward). "
            "Rename your bathymetry variable to 'deptho'."
        )

    ds_in = ds[["lon", "lat"]] if engine == "xesmf" else None

    vars = [var] if isinstance(var, (str, bytes)) else list(var)
    X = np.asarray(transect_length, dtype=float)

    # (1) Boundary contour and (2) resampled sections along it
    contour_xy, contour_lon, contour_lat = longest_boundary_contour(ds, boundary_mask, crs=crs)
    contour_xy, s_m = _resample_contour_xy(contour_xy, transect_spacing)
    _, n = _tangent_normal_xy(contour_xy)
    normal_sign = _contour_outward_sign(contour_xy)

    x0, y0 = contour_xy[:, 0], contour_xy[:, 1]
    nx, ny = n[:, 0].copy() * normal_sign, n[:, 1].copy() * normal_sign

    tf_inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    deptho_filled = ds["deptho"].fillna(0.0)

    # (3) Final transects
    lon_t, lat_t = _make_transect_lonlat(tf_inv, x0, y0, nx, ny, X)

    # (6) Sample requested variables
    obj_vars = [vn for vn in vars if vn != "deptho"]
    obj = ds[obj_vars + ["deptho"]].where(deptho_filled > 0)
    vloc = _sample_vars(
        ds, ds_in, obj, lon_t, lat_t,
        engine=engine, method=method, regridder=None,
        lon_name=lon_name, lat_name=lat_name,
    )
    vloc = _unstack_loc(vloc, X, s_m)
    dloc = vloc["deptho"].values
    wet = np.isfinite(dloc) & (dloc > 0)
    vloc = vloc.where(wet)
    if "deptho" not in vars:
        vloc = vloc.drop_vars("deptho")
    vloc = vloc.assign_coords(
        depth=(
            ("section", "transect_length"),
            np.where(wet, dloc, np.nan),
            {"units": "m", "description": "Sampled depth along transects."},
        ),
        lon=(
            ("section", "transect_length"),
            lon_t,
            {"units": "degrees_east", "description": "Transect longitude."},
        ),
        lat=(
            ("section", "transect_length"),
            lat_t,
            {"units": "degrees_north", "description": "Transect latitude."},
        ),
    )

    xshelf = vloc[vars[0]] if len(vars) == 1 else vloc
    if not return_geometry:
        return xshelf

    lon0, lat0 = tf_inv.transform(x0, y0)
    geometry = _build_transect_geometry_dataset(
        lon0=lon0,
        lat0=lat0,
        nx=nx,
        ny=ny,
        dloc=dloc,
        contour_lon=contour_lon,
        contour_lat=contour_lat,
        s_m=s_m,
        transect_length=X,
        crs=crs,
        engine=engine,
        method=method,
        transect_spacing=transect_spacing,
    )

    return xshelf, geometry
