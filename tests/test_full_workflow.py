import numpy as np
import xarray as xr
import pytest

from xshelftransects import cross_shelf_transects, longest_boundary_contour


def _gaussian_bathymetry(lon2d, lat2d, center=(2.0, 2.0), sigma=0.8):
    # Smooth Gaussian field used for bathymetry and derived boundary masks.
    cx, cy = center
    return np.exp(-(((lon2d - cx) ** 2 + (lat2d - cy) ** 2) / (2.0 * sigma**2)))


def _toy_dataset():
    # Gaussian bathymetry with a known center and decay scale.
    lon1d = np.linspace(0.0, 4.0, 9)
    lat1d = np.linspace(0.0, 4.0, 9)
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    gauss = _gaussian_bathymetry(lon2d, lat2d)
    deptho = 1000.0 + 400.0 * gauss
    temp = lon2d + lat2d
    return xr.Dataset(
        {
            "lon": (("lat1d", "lon1d"), lon2d),
            "lat": (("lat1d", "lon1d"), lat2d),
            "deptho": (("lat1d", "lon1d"), deptho),
            "temp": (("lat1d", "lon1d"), temp),
        }
    ).assign_coords(lon1d=("lon1d", lon1d), lat1d=("lat1d", lat1d))


def test_longest_boundary_contour_centered_on_gaussian():
    pytest.importorskip("contourpy")
    ds = _toy_dataset()
    # Normalize bathymetry to a 0-1 field, then threshold for a simple boundary mask.
    # Normalize bathymetry to 0-1, then threshold to build a simple Gaussian-derived mask.
    # Normalize bathymetry to 0-1, then threshold to build a simple Gaussian-derived mask.
    deptho = ds["deptho"].values
    gauss = (deptho - deptho.min()) / (deptho.max() - deptho.min())
    boundary_mask = xr.DataArray((gauss >= 0.5).astype(int), dims=("lat1d", "lon1d"))
    _, contour_lon, contour_lat = longest_boundary_contour(
        ds, boundary_mask, crs="EPSG:3857"
    )
    # Contour mean should remain near the Gaussian center in lon/lat.
    assert np.isclose(contour_lon.mean(), 2.0, atol=0.5)
    assert np.isclose(contour_lat.mean(), 2.0, atol=0.5)


def test_cross_shelf_transects_matches_plane_field():
    pytest.importorskip("contourpy")
    ds = _toy_dataset()
    # Same Gaussian-derived mask as above to keep the contour centered.
    deptho = ds["deptho"].values
    gauss = (deptho - deptho.min()) / (deptho.max() - deptho.min())
    boundary_mask = xr.DataArray((gauss >= 0.5).astype(int), dims=("lat1d", "lon1d"))
    X = np.array([0.0, 50e3, 100e3])

    # Full pipeline: expected values follow temp = lon + lat plane.
    xshelf, geometry = cross_shelf_transects(
        ds,
        var="temp",
        boundary_mask=boundary_mask,
        transect_length=X,
        transect_spacing=100e3,
        crs="EPSG:3857",
        engine="xarray",
        method="bilinear",
        lon_name="lon1d",
        lat_name="lat1d",
        return_geometry=True,
    )
    # Transect_length should round-trip to the output coordinates.
    assert xshelf.sizes["transect_length"] == X.size
    # Geometry depth should align with section/transect_length.
    assert geometry["depth_xshelf"].shape == (
        xshelf.sizes["section"],
        X.size,
    )
    expected = xshelf["lon"].values + xshelf["lat"].values
    # temp field is lon+lat, so interpolation should match exactly.
    assert np.allclose(xshelf.values, expected, equal_nan=True)
    # The Gaussian-derived contour should stay centered near (2, 2) in lon/lat.
    assert np.isclose(geometry["contour_lon"].mean().item(), 2.0, atol=0.5)
    assert np.isclose(geometry["contour_lat"].mean().item(), 2.0, atol=0.5)
