from __future__ import annotations

__all__ = ["ISOPVirtualStack", "save_all_volumes_to_tiff"]

import os
import struct
from pathlib import Path
from typing import Final

import numpy as np
import tifffile


class ISOPVirtualStack:
    """Memory-mapped (T, C, Z, Y, X) big-endian uint16 view of a .3DU16 file.

    Attributes
    ----------
    shape : (T, C, Z, Y, X)
    scales : (z, y, x) voxel size
    data : zero-copy (T, C, Z, Y, X) >u2 view
    volume_headers : zero-copy (T, 4) >u4 view of the per-volume headers

    The mapping is released automatically when this object and every
    array taken from it are garbage-collected.
    """

    _GLOBAL_HEADER: Final = 52  # bytes before the first volume
    _VOL_HEADER: Final = 16  # bytes before each volume (4 x big-endian uint32)

    def __init__(self, filepath: os.PathLike | str):
        self.filepath = Path(filepath)

        with self.filepath.open("rb") as f:
            raw = f.read(self._GLOBAL_HEADER)
        try:
            # skip bytes 0-3 and 28-31
            scale_z, scale_y, scale_x, T, C, Z, Y, X = struct.unpack(">4x3d4x5i", raw)
        except struct.error as e:
            raise ValueError(f"Not a valid 3DU16 file: {self.filepath}") from e

        self.scales = (scale_z, scale_y, scale_x)
        self.shape = (T, C, Z, Y, X)

        record = np.dtype(
            [
                ("hdr", ">u4", (self._VOL_HEADER // 4,)),
                ("data", ">u2", (C, Z, Y, X)),
            ]
        )
        expected = self._GLOBAL_HEADER + T * record.itemsize
        actual = self.filepath.stat().st_size
        if actual < expected:
            raise ValueError(
                f"File is truncated: expected at least {expected} bytes, got {actual}"
            )

        mm = np.memmap(
            self.filepath,
            dtype=record,
            mode="r",
            offset=self._GLOBAL_HEADER,
            shape=(T,),
        )
        self.data: np.ndarray = mm["data"]
        self.volume_headers: np.ndarray = mm["hdr"]

    def __repr__(self) -> str:
        return (
            f"ISOPVirtualStack({str(self.filepath)!r}, "
            f"shape={self.shape}, scales={self.scales})"
        )

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key) -> np.ndarray:
        """Unsigned int16, Big Endian, vs[t] -> (C, Z, Y, X); supports slices, negatives, vs[t, c], ..."""
        return self.data[key]

    def volume(self, t: int) -> np.ndarray:
        """Copy of volume t as native-endian uint16 (C, Z, Y, X)."""
        return self.data[t].astype(np.uint16)


def save_all_volumes_to_tiff(
    stack: ISOPVirtualStack,
    filename: os.PathLike | str,
):
    z_scale, y_scale, x_scale = stack.scales
    T, C, Z, Y, X = stack.shape
    tifffile.imwrite(
        filename,
        (stack.volume(t).swapaxes(0, 1) for t in range(T)),  # CZYX -> ZCYX
        shape=stack.shape,
        dtype="u2",
        imagej=True,
        resolution=(1.0 / x_scale, 1.0 / y_scale),
        metadata={
            "spacing": z_scale,
            "unit": "um",
            "axes": "TZCYX",
        },
        photometric="minisblack",
    )
