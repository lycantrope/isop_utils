"""Memory-mapped reader for ISOP ``.3DU16`` time-lapse volume files.

File layout (all values big-endian)::

    global header   52 bytes   4 skipped, scale z/y/x (3 x f8), 4 skipped,
                               T, C, Z, Y, X (5 x i4)
    T records, each:
        volume header   16 bytes   4 x u4
        volume data     C*Z*Y*X x u2, stored in (C, Z, Y, X) order

Example
-------
>>> vs = ISOPVirtualStack("sample.3DU16")
>>> vs.shape                 # (T, C, Z, Y, X)
>>> vol = vs.volume(0)       # native uint16 copy, (C, Z, Y, X)
>>> isop_3du16_to_tiff(vs, "sample.tif")
"""

from __future__ import annotations

__all__ = ["ISOPVirtualStack", "isop_3du16_to_tiff"]

import os
import struct
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray


class ISOPVirtualStack:
    """Zero-copy, memory-mapped view of a ``.3DU16`` file.

    The whole file is mapped once, read-only, with a structured dtype
    (volume header + volume data per frame). Nothing is read from disk
    until the data is accessed.

    Parameters
    ----------
    filepath
        Path to the ``.3DU16`` file.

    Attributes
    ----------
    filepath : Path
        Path to the file.
    shape : tuple of int
        ``(T, C, Z, Y, X)``.
    scales : tuple of float
        Voxel size ``(z, y, x)``, in micrometres.
    data : ndarray, shape (T, C, Z, Y, X), dtype >u2
        Zero-copy big-endian view of all volumes.
    volume_headers : ndarray, shape (T, 4), dtype >u4
        Zero-copy view of the per-volume headers.

    Raises
    ------
    ValueError
        If the global header cannot be parsed, or the file is shorter
        than the header says it should be.

    Notes
    -----
    There is no ``close()``. The mapping is released automatically once
    this object and every array taken from it have been garbage-collected,
    so returned arrays stay valid after the reader is gone.
    """

    _GLOBAL_HEADER: Final = 52
    """Bytes before the first volume record."""

    _VOL_HEADER: Final = 16
    """Bytes of header before each volume (4 x big-endian uint32)."""

    filepath: Path
    shape: tuple[int, int, int, int, int]
    scales: tuple[float, float, float]
    data: NDArray[np.uint16]
    volume_headers: NDArray[np.uint32]

    def __init__(self, filepath: os.PathLike[str] | str) -> None:
        self.filepath = Path(filepath).resolve()

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
        self.data = mm["data"]
        self.volume_headers = mm["hdr"]

    def __repr__(self) -> str:
        return (
            f"ISOPVirtualStack({str(self.filepath)!r}, "
            f"shape={self.shape}, scales={self.scales})"
        )

    def __reduce__(self) -> tuple[type[ISOPVirtualStack], tuple[str]]:
        # Pickle only the path; the receiving process re-parses the header and re-maps.
        return (self.__class__, (str(self.filepath),))

    def __len__(self) -> int:
        """Number of time points ``T``."""
        return self.shape[0]

    def __getitem__(self, key: Any) -> NDArray[np.uint16]:
        """Index :attr:`data` with any NumPy index; returns a zero-copy view.

        ``vs[t]`` gives ``(C, Z, Y, X)``. Slices, negative indices and
        multi-axis indices such as ``vs[t, c]`` or ``vs[2:5, :, 10]`` also
        work. The result is big-endian ``>u2``; use :meth:`volume` for a
        native-endian copy.
        """
        result: NDArray[np.uint16] = self.data[key]
        return result

    def volume(self, t: int) -> NDArray[np.uint16]:
        """Return volume ``t`` as a native-endian uint16 copy.

        Parameters
        ----------
        t
            Time index. Negative values count from the end.

        Returns
        -------
        ndarray, shape (C, Z, Y, X), dtype uint16
            C-contiguous array in the machine's native byte order, ready
            for libraries that reject big-endian data (PyTorch, Numba, ...).
        """
        result: NDArray[np.uint16] = self.data[t].astype(np.uint16)
        return result


def isop_3du16_to_tiff(
    stack: ISOPVirtualStack,
    filename: os.PathLike[str] | str,
) -> None:
    """Write every volume to an ImageJ hyperstack TIFF.

    Volumes are read and written one at a time, so memory use stays at
    about one volume regardless of file size.

    Parameters
    ----------
    stack
        Source stack.
    filename
        Output ``.tif`` path. Overwritten if it exists.

    Notes
    -----
    The output axis order is ``TZCYX`` (ImageJ's order), with uint16
    pixels, XY resolution and Z spacing taken from ``stack.scales`` in
    micrometres. Requires ``tifffile``.
    """
    import tifffile

    z_scale, y_scale, x_scale = stack.scales
    T, C, Z, Y, X = stack.shape
    tifffile.imwrite(
        filename,
        (stack.volume(t).swapaxes(0, 1) for t in range(T)),  # CZYX -> ZCYX
        shape=(T, Z, C, Y, X),
        dtype=np.uint16,
        imagej=True,
        resolution=(1.0 / x_scale, 1.0 / y_scale),
        metadata={"spacing": z_scale, "unit": "um", "axes": "TZCYX"},
        photometric="minisblack",
    )
