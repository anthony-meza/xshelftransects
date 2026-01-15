xshelftransects
==============

Build and sample cross-shelf transects from a binary boundary mask on model grids. Useful for understanding gravity current and coastally-trapped waves, which are defined by their orientation and proximity to the continental shelves. 

Install
-------
```bash
conda create --name xshelftransects python=3.11
conda activate xshelftransects
pip install -e .
```

Optional dependencies
---------------------
- xesmf (for `engine="xesmf"` on curvilinear grids)
- scipy (for Theil-Sen based transect orientation)

Basic usage
-----------
```python
import numpy as np
import xshelftransects

transects, geometry = xshelftransects.cross_shelf_transects(
    ds,
    "thetao",
    boundary_mask,  # binary mask: 0/1 interface is transect_length=0
    transect_length=np.arange(0.0, 200e3 + 2e3, 2e3),
    transect_spacing=10e3,
    crs="EPSG:3031",
    engine="xesmf",
)
```

Inputs and expectations
- `ds["lon"]`, `ds["lat"]`: 2D longitude/latitude arrays.
- `ds["deptho"]`: ocean depth in meters, positive downward (deptho <= 0 treated as land/ice).
- `boundary_mask`: DataArray on the same grid as lon/lat; 0/1 interface is the boundary.
- `crs`: projected CRS in meters (e.g., "EPSG:3031" Antarctica, "EPSG:3413" Arctic,
  "EPSG:3857" global).

Outputs
- `transects`: sampled values with dims (..., section, transect_length) and lon/lat coords.
- `geometry`: Dataset with anchor points, normals, and boundary contour.
