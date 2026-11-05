# %%
from pathlib import Path

import numpy as np
import tifffile

p = Path(
    "//FS/kuan/Confocal/250617_AML320_AML318/tiff/250617_NeuroPAL_AML320_G5_001.tif"
)

rawdata = tifffile.memmap(p)

# Apply median filter along y axis (kernel=3)
Z, C = rawdata.shape[:2]


for z in range(Z):
    for c in range(C):
        im = rawdata[z, c]
        im = np.pad(im, ((1, 1), (0, 0)), mode="edge")
        im_view = np.lib.stride_tricks.sliding_window_view(im, 3, 0)
        im_median = np.median(im_view, axis=-1)
        rawdata[z, c] = im_median
        rawdata.flush()
del rawdata


# %%
# %%
# Revere Z order

p = Path("//FS/kuan/Confocal/250618_AML320/tiff/250618_NeuroPAL_AML320_G4_001.tif")

rawdata = tifffile.memmap(p)


rawdata[...] = rawdata[::-1]

rawdata.flush()
del rawdata
# %%
