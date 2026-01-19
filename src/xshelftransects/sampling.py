import numpy as np
import xarray as xr
import pandas as pd

try:
    import xesmf as xe
except ModuleNotFoundError:
    xe = None


# =============================================================================
# Sampling helpers
# =============================================================================

def _locstream_out(lon_t, lat_t):
    """
    Build a LocStream output grid (1D 'loc') from (section, transect_length) lon/lat arrays.
    """
    return xr.Dataset(
        {"lon": xr.DataArray(np.asarray(lon_t).ravel(), dims=("loc",)),
         "lat": xr.DataArray(np.asarray(lat_t).ravel(), dims=("loc",))}
    )


def _as_dataset(obj):
    """
    Normalize a DataArray/Dataset into a Dataset so multi-variable sampling is uniform.

    Returns
    -------
    xarray.Dataset
    """
    if isinstance(obj, xr.Dataset):
        return obj
    if isinstance(obj, xr.DataArray):
        name = obj.name if obj.name is not None else "var"
        return obj.to_dataset(name=name)
    raise TypeError("obj must be xarray.DataArray or xarray.Dataset")


def _sample_vars_xesmf(ds_in, obj, lon_t, lat_t, method, regridder=None):
    """
    xESMF-based sampling at LocStream target points (optional dependency).

    Parameters
    ----------
    ds_in : xarray.Dataset
        Must contain ds_in["lon"], ds_in["lat"] defining the source grid (typically 2D).
    obj : xarray.DataArray or xarray.Dataset
        Field(s) on the source grid to sample.
    lon_t, lat_t : ndarray
        Target lon/lat arrays with shape (section, transect_length).
    method : str
        xESMF method, e.g. "bilinear" or "nearest_s2d".
    regridder : xesmf.Regridder or None
        If provided, used directly.

    Returns
    -------
    xarray.Dataset
        Sampled output with trailing dimension "loc".
    """
    if xe is None:
        raise ModuleNotFoundError(
            "engine='xesmf' requires the optional dependency 'xesmf'. "
            "Install it (and its ESMF backend) to use xESMF-based sampling."
        )

    ds_out = _locstream_out(lon_t, lat_t)
    if regridder is None:
        regridder = xe.Regridder(
            ds_in,
            ds_out,
            method,
            locstream_out=True,
            unmapped_to_nan=True,
        )
    return regridder(_as_dataset(obj))


def _sample_vars_xarray(obj, lon_t, lat_t, method, lon_name="lon", lat_name="lat"):
    """
    xarray.interp-based sampling at LocStream target points.

    Important limitation
    --------------------
    This only applies to rectilinear grids where `obj` has 1D lon/lat coordinates
    (tensor-product grid). It is not a curvilinear regridder.

    method mapping
    --------------
    - method="bilinear" -> xarray method="linear"

    Returns
    -------
    xarray.Dataset
        Sampled output with trailing dimension "loc".
    """
    ds_out = _locstream_out(lon_t, lat_t)
    interp_method = "linear" if method == "bilinear" else method

    out = xr.Dataset()
    lon_loc = ds_out["lon"]
    lat_loc = ds_out["lat"]
    for vn, da in _as_dataset(obj).data_vars.items():
        out[vn] = da.interp({lon_name: lon_loc, lat_name: lat_loc}, method=interp_method)
    return out


def _sample_vars(
    ds,
    ds_in,
    obj,
    lon_t,
    lat_t,
    engine="xesmf",
    method="bilinear",
    regridder=None,
    lon_name="lon",
    lat_name="lat",
):
    """
    Dispatch sampling to either xESMF or xarray.

    Returns
    -------
    xarray.Dataset with trailing dimension "loc".
    """
    if engine == "xesmf":
        return _sample_vars_xesmf(
            ds_in, obj, lon_t, lat_t, method,
            regridder=regridder,
        )
    if engine == "xarray":
        return _sample_vars_xarray(obj, lon_t, lat_t, method, lon_name=lon_name, lat_name=lat_name)
    raise ValueError("engine must be 'xesmf' or 'xarray'")


def _unstack_loc(vloc_ds, transect_length, s_m):
    """
    Convert (..., loc) output back to (..., section, transect_length) using a MultiIndex.

    Parameters
    ----------
    vloc_ds : xarray.Dataset
        Output from _sample_vars with a trailing 'loc' dimension.
    transect_length : (nx,) ndarray
        Cross-shelf coordinate values.
    s_m : (nsec,) ndarray
        Along-boundary distances for sections.

    Returns
    -------
    xarray.Dataset
        Same data with dimensions (..., section, transect_length), plus coord s_m(section).
    """
    nsec = s_m.size
    # MultiIndex flattens the (section, transect_length) grid into a 1D "loc" index.
    mi = pd.MultiIndex.from_product([np.arange(nsec), transect_length], names=("section", "transect_length"))
    # Explicitly wrap the MultiIndex to keep xarray's coordinate behavior stable.
    mindex_coords = xr.Coordinates.from_pandas_multiindex(mi, "loc")
    return (
        vloc_ds
        # The loc dimension is a flattened (section, transect_length) grid.
        .assign_coords(mindex_coords)
        # Unstack loc back into 2D section/transect_length dimensions.
        .unstack("loc")
        # Keep along-boundary distance as a section coordinate.
        .assign_coords(s_m=("section", np.asarray(s_m)))
    )
