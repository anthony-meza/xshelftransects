import numpy as np
import xarray as xr

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
    - The normal sign is arbitrary here; it is oriented later so that depth increases with +transect_length.
    """
    dxy = np.gradient(np.asarray(contour_xy), axis=0)
    t = dxy / np.maximum(np.linalg.norm(dxy, axis=1, keepdims=True), 1e-12)
    n = np.c_[-t[:, 1], t[:, 0]]
    return t, n


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


def _orient_normals_by_depth_polyfit(transect_length, dloc0):
    """
    Orient normals so that +transect_length points offshore (depth increases with transect_length).

    Mechanism
    ---------
    - Given depth sampled along provisional transects (dloc0[section, transect_length]),
      fit depth(transect_length) with a 1st-degree polynomial per section.
    - If slope < 0, flip the normal for that section.

    Returns
    -------
    flip : (nsec,) ndarray of {+1, -1}
    """
    depth_tran0 = xr.DataArray(
        dloc0,
        dims=("section", "transect_length"),
        coords={"section": np.arange(dloc0.shape[0]), "transect_length": transect_length},
    ).where(lambda z: z > 0)

    pf = depth_tran0.polyfit(dim="transect_length", deg=1, skipna=True)
    coef = pf["polyfit_coefficients"]
    slope = coef.sel(degree=1) if 1 in coef["degree"].values else coef.isel(degree=0)
    return xr.where(slope < 0, -1.0, 1.0).values


def _build_transect_geometry_dataset(
    lon0,
    lat0,
    nx,
    ny,
    lon_t,
    lat_t,
    dloc,
    contour_lon,
    contour_lat,
    s_m,
    transect_length,
    crs,
    engine,
    method,
    reuse_weights,
    transect_spacing,
):
    """
    Build the geometry dataset for cross-shelf transects.
    """
    return xr.Dataset(
        data_vars=dict(
            lon=(("section", "transect_length"), lon_t, {"units": "degrees_east", "description": "Transect longitude."}),
            lat=(("section", "transect_length"), lat_t, {"units": "degrees_north", "description": "Transect latitude."}),
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
            reuse_weights=bool(reuse_weights),
            description=(
                "Cross-shelf transects built from boundary_mask contour; transect_length=0 at "
                "contour; +transect_length oriented toward deeper water; land masked to NaN."
            ),
            deptho_convention="deptho is ocean depth in meters, positive downward; deptho<=0 treated as land/ice.",
            transect_spacing=float(transect_spacing),
            optional_dependency="xesmf is only required when engine='xesmf'",
        ),
    )
