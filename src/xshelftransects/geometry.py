import warnings
import numpy as np
import xarray as xr

try:
    from scipy.stats import theilslopes
except ModuleNotFoundError:
    theilslopes = None

def _project_lonlat(lon2d, lat2d, crs_out):
    """
    Project 2D lon/lat arrays to x/y in meters for a specified projected CRS.

    Parameters
    ----------
    lon2d, lat2d : array-like
        2D longitude/latitude arrays (degrees).
    crs_out : str
        Projected CRS (meters), e.g., "EPSG:3031" for Antarctica, "EPSG:3413" for
        the Arctic, or "EPSG:3857" for a global web-mercator projection.

    Returns
    -------
    x2d, y2d : ndarray
        Projected coordinates in meters.
    """
    from pyproj import Transformer

    # Build a lon/lat (EPSG:4326) -> projected CRS transformer; EPSG:4326 is degrees,
    # so projected outputs are in meters when crs_out is a meter-based CRS (e.g., EPSG:3031).
    tf = Transformer.from_crs("EPSG:4326", crs_out, always_xy=True)
    return tf.transform(np.asarray(lon2d), np.asarray(lat2d))


def _resample_contour_xy(contour_xy, transect_spacing):
    """
    Resample a contour to approximately uniform spacing along arc-length.

    This is used to generate evenly spaced "sections" along a predefined contour. 

    Parameters
    ----------
    contour_xy : (N,2) ndarray
        Contour vertices in projected meters.
    transect_spacing : float
        Target along-contour spacing (meters).

    Returns
    -------
    contour_xy_rs : (M,2) ndarray
        Resampled contour vertices in projected meters.
    s_m : (M,) ndarray
        Along-contour distance from start (meters) for each resampled vertex.
    """
    contour_xy = np.asarray(contour_xy)
    seg = np.sqrt(np.sum(np.diff(contour_xy, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return contour_xy, s
    n = max(2, int(np.floor(s[-1] / transect_spacing)) + 1)
    si = np.linspace(0.0, s[-1], n)
    xi = np.interp(si, s, contour_xy[:, 0])
    yi = np.interp(si, s, contour_xy[:, 1])
    return np.c_[xi, yi], si


def _tangent_normal_xy(contour_xy):
    """
    Unit tangents and normals along a contour line in Cartesian coordinates.

    Notes
    -----
    - np.gradient is taken with respect to vertex index, not an explicit spatial coordinate.
      Because the contour is resampled to roughly uniform spacing and we normalize to unit
      vectors, this is typically adequate.
    - The normal sign is arbitrary here; it is oriented later using contour orientation.
    """
    dxy = np.gradient(np.asarray(contour_xy), axis=0)
    t = dxy / np.maximum(np.linalg.norm(dxy, axis=1, keepdims=True), 1e-12)
    n = np.c_[-t[:, 1], t[:, 0]]
    return t, n


def _contour_outward_sign(contour_xy):
    """
    Determine a global sign (+1 or -1) for outward normals based on contour orientation.

    Notes
    -----
    - For a counter-clockwise contour, the left normal points inward, so outward is -1.
    - For a clockwise contour, the left normal points outward, so outward is +1.
    """
    contour_xy = np.asarray(contour_xy)
    if contour_xy.shape[0] < 3:
        return 1.0
    x = contour_xy[:, 0]
    y = contour_xy[:, 1]
    area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    if not np.isfinite(area) or area == 0.0:
        return 1.0
    return -1.0 if area > 0.0 else 1.0


def _make_transect_lonlat(tf_inv, x0, y0, nx, ny, X):
    """
    Create transect target points (lon/lat) from projected anchors and unit normals.

    Each transect is a straight line:
        (x,y)(section, transect_length) = (x0,y0)(section) + transect_length * (nx,ny)(section)

    Returns
    -------
    lon_t, lat_t : (nsec, nx) ndarrays
        Target lon/lat coordinates for sampling.
    """
    xt = x0[:, None] + nx[:, None] * X[None, :]
    yt = y0[:, None] + ny[:, None] * X[None, :]
    lon_t, lat_t = tf_inv.transform(xt, yt)
    return np.asarray(lon_t), np.asarray(lat_t)


def _orient_normals_by_offshore_slope(transect_length, dloc0, offshore_slope_window=200e3, min_wet_fraction=0.5):
    """
    Orient normals so that +transect_length points offshore (depth increases with transect_length).

    Mechanism
    ---------
    - Given depth sampled along provisional transects (dloc0[section, transect_length]),
      check that near-boundary points are wet, then estimate the near-boundary slope
      from points within offshore_slope_window of the boundary.
    - If the median slope < 0, flip the normal for that section.

    Returns
    -------
    flip : (nsec,) ndarray of {+1, -1}
    """
    depth_tran0 = xr.DataArray(
        dloc0,
        dims=("section", "transect_length"),
        coords={"section": np.arange(dloc0.shape[0]), "transect_length": transect_length},
    ).where(lambda z: z > 0)

    x = depth_tran0["transect_length"]
    window_mask = x <= offshore_slope_window
    wet_mask = xr.apply_ufunc(np.isfinite, depth_tran0).astype(bool)
    wet_first = wet_mask.where(window_mask).sum("transect_length")
    n_window = window_mask.sum().item()
    min_wet = max(1, int(np.ceil(min_wet_fraction * n_window)))

    dx = xr.DataArray(
        np.diff(np.asarray(transect_length)),
        dims=("transect_length",),
        coords={"transect_length": depth_tran0["transect_length"].isel(transect_length=slice(0, -1))},
    )
    depth_filled = depth_tran0.fillna(0.0)
    slopes = depth_filled.diff("transect_length") / dx
    valid_pair = wet_mask & wet_mask.shift(transect_length=-1, fill_value=False)
    slopes = slopes.where(valid_pair.isel(transect_length=slice(0, -1)))
    slopes_window = slopes.where(window_mask.isel(transect_length=slice(0, -1)))
    slope_med = slopes_window.median("transect_length", skipna=True)

    def _theil_sen_1d(y, x_vals):
        if theilslopes is None:
            raise ModuleNotFoundError(
                "Theil-Sen slope requires scipy. Install scipy or choose a different orientation method."
            )
        valid = np.isfinite(y)
        if valid.sum() < 2:
            warnings.warn("Theil-Sen slope: fewer than 2 valid points; returning NaN.", RuntimeWarning)
            return np.nan
        xi = x_vals[valid]
        yi = y[valid]
        slope, _, _, _ = theilslopes(yi, xi)
        return slope

    depth_window = depth_tran0.where(window_mask)
    slope_ts = xr.apply_ufunc(
        lambda y: _theil_sen_1d(y, np.asarray(transect_length)),
        depth_window,
        input_core_dims=[["transect_length"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    flip = xr.where(wet_first < min_wet, -1.0, 1.0)
    flip = xr.where(slope_med < 0, -1.0, flip)
    flip = xr.where(slope_ts < 0, -1.0, flip)
    return flip.values


def _build_transect_geometry_dataset(
    lon0,
    lat0,
    nx,
    ny,
    dloc,
    contour_lon,
    contour_lat,
    s_m,
    transect_length,
    crs,
    engine,
    method,
    transect_spacing,
):
    """
    Build the geometry dataset for cross-shelf transects.
    """
    return xr.Dataset(
        data_vars=dict(
            lon0=(("section",), np.asarray(lon0), {"units": "degrees_east", "description": "Longitude where transect intersects boundary (transect_length=0)."}),
            lat0=(("section",), np.asarray(lat0), {"units": "degrees_north", "description": "Latitude where transect intersects boundary (transect_length=0)."}),
            s_m=(("section",), np.asarray(s_m), {"units": "m", "description": "Along-boundary distance from start."}),
            nx=(("section",), np.asarray(nx), {"units": "1", "description": "Unit normal x-component in projected CRS."}),
            ny=(("section",), np.asarray(ny), {"units": "1", "description": "Unit normal y-component in projected CRS."}),
            depth_xshelf=(("section", "transect_length"), dloc, {"units": "m", "description": "Sampled depth along transects."}),
            contour_lon=(("contour_pt",), np.asarray(contour_lon), {"units": "degrees_east", "description": "Boundary contour longitude."}),
            contour_lat=(("contour_pt",), np.asarray(contour_lat), {"units": "degrees_north", "description": "Boundary contour latitude."}),
        ),
        coords=dict(
            section=("section", np.arange(s_m.size), {"description": "Section index along boundary."}),
            transect_length=("transect_length", transect_length, {"units": "m", "description": "Cross-shelf distance from boundary."}),
            contour_pt=("contour_pt", np.arange(np.asarray(contour_lon).size), {"description": "Boundary contour vertex index."}),
        ),
        attrs=dict(
            crs=crs,
            engine=engine,
            sampling_method=method,
            description=(
                "Cross-shelf transects built from boundary_mask contour; transect_length=0 at "
                "contour; +transect_length oriented toward deeper water; land masked to NaN."
            ),
            deptho_convention="deptho is ocean depth in meters, positive downward; deptho<=0 treated as land/ice.",
            transect_spacing=float(transect_spacing),
            optional_dependency="xesmf is only required when engine='xesmf'",
        ),
    )
