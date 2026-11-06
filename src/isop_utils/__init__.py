from __future__ import annotations

__all__ = [
    "ISOPHeader",
    "ISOPVirtualStack",
]

import mmap
import os
import struct
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from warnings import warn

import h5py
import hdf5plugin
import numpy as np
import tifffile

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray

LUT_TABLE: dict[str, np.ndarray] = {
    "u1": np.floor(np.arange(65536) / 256).astype(np.uint8),
    "u2": np.arange(65536),
    "f4": np.linspace(0.0, 1.0, 65536, endpoint=True, dtype=np.float32),
}


class ISOPHeader(NamedTuple):
    x_scale: float
    y_scale: float
    z_scale: float
    n_frame: int
    n_channel: int
    depth: int
    height: int
    width: int

    @classmethod
    def from_3DU16(cls, filename: os.PathLike | str) -> ISOPHeader:
        filename = Path(filename)
        if filename.suffix.lower() != ".3du16":
            raise TypeError(f"Only support (*.3d16 or *.3DU16): {filename.name}")
        try:
            # skip bytes from 0 to 3 and 28-31
            with filename.open("rb") as fd:
                HEADER = fd.read(52)
            (z_scale, y_scale, x_scale, T, C, Z, Y, X) = struct.unpack(
                ">4x3d4x5i", HEADER
            )
        except struct.error as e:
            raise TypeError(f"This file might not be a valid 3DU16 file: {e}")

        return cls(x_scale, y_scale, z_scale, T, C, Z, Y, X)

    @property
    def bytes_per_xy(self) -> int:
        return self.height * self.width * 2

    @property
    def bytes_per_xyz(self) -> int:
        return self.height * self.width * self.depth * 2

    @property
    def pixels_per_volume(self) -> int:
        return self.height * self.width * self.depth * self.n_channel

    @property
    def bytes_per_volume(self) -> int:
        gap = 16
        return self.pixels_per_volume * 2 + gap

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        # TCZYX
        return (self.n_frame, self.n_channel, self.depth, self.height, self.width)

    @property
    def scales(self) -> tuple[float, float, float]:
        # z, y, x
        return (self.z_scale, self.y_scale, self.x_scale)


class ISOPVirtualStack:
    def __init__(self, filepath: os.PathLike):
        self.filepath = Path(filepath)
        self.header = ISOPHeader.from_3DU16(self.filepath)

        self.raf = None
        self.handler = None

        self._cleanup = ExitStack()

    def __repr__(self) -> str:
        return f"ISOPVitrualStack(filepath={self.filepath!s}, header={self.header!s})"

    def __enter__(self):
        self._cleanup.__enter__()
        try:
            self.handler = self._cleanup.enter_context(open(self.filepath, "rb"))
            self.raf = self._cleanup.enter_context(
                mmap.mmap(self.handler.fileno(), 0, access=mmap.ACCESS_READ)
            )
        except Exception:
            # もし初期化の途中でエラーが起きたら、開いた分を即座にクリーンアップして例外を再送
            self._cleanup.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._cleanup.__exit__(exc_type, exc_val, exc_tb)

    def get_volume(
        self,
        index: int,
        dtype: Literal["u1", "f4", "u2"] = "u2",
        imagej: bool = False,
    ) -> NDArray[np.uint8] | NDArray[np.uint16] | NDArray[np.float32]:
        if self.raf is None:
            raise RuntimeError(
                f"Please run {self.__class__.__name__} under the 'with statement'"
            )
        T, C, Z, Y, X = self.header.shape
        bytes_per_volume = self.header.bytes_per_volume
        pixels_per_volume = self.header.pixels_per_volume

        if (index < 0) | (index >= T):
            raise IndexError(f"Index is out of boundary: {index}")

        gap = 16
        offset = 52 + index * bytes_per_volume

        stack = (
            np.frombuffer(
                self.raf,
                dtype=">u2",
                count=pixels_per_volume,
                offset=offset + gap,
            )
            .reshape((C, Z, Y, X))
            .copy()
        )

        # CZYX -> TCZYX
        stack = stack[np.newaxis, :, :, :, :]

        if imagej:
            # TCZYX -> TZCYX
            stack = stack.transpose((0, 2, 1, 3, 4))

        # type conversion
        if dtype not in LUT_TABLE:
            warn(
                UserWarning(
                    f"Unsupported conversion: {dtype}, output the original format (uint16)"
                )
            )
            dtype = "u2"

        if dtype == "u2":
            return stack

        return LUT_TABLE[dtype][stack]

    def get_multi_volumes(
        self,
        start: int,
        end: int,
        dtype: Literal["u1", "f4", "u2"] = "u2",
        imagej: bool = False,
    ) -> NDArray[np.uint8] | NDArray[np.uint16] | NDArray[np.float32]:
        if self.raf is None:
            raise RuntimeError(
                f"Please run {self.__class__.__name__} under the 'with statement'"
            )
        T, C, Z, Y, X = self.header.shape
        bytes_per_volume = self.header.bytes_per_volume

        if (start < 0) | (start >= T):
            raise IndexError(f"Index is out of boundary: start={start}")

        if (end < 0) | (end >= T):
            raise IndexError(f"Index is out of boundary: end={end}")

        if start > end:
            start, end = end, start

        T = end - start + 1

        gap = 16
        offset = 52 + start * bytes_per_volume

        stack = (
            np.frombuffer(
                self.raf,
                dtype=">u2",
                count=(bytes_per_volume * T) // 2,
                offset=offset,
            )
            .reshape((T, -1))[:, gap // 2 :]
            .reshape((T, C, Z, Y, X))
            .copy()
        )

        if imagej:
            # TCZYX -> TZCYX
            stack = stack.transpose((0, 2, 1, 3, 4))

        # type conversion
        if dtype not in LUT_TABLE:
            warn(
                UserWarning(
                    f"Unsupported conversion: {dtype}, output the original format (uint16)"
                )
            )
            dtype = "u2"

        if dtype == "u2":
            return stack

        return LUT_TABLE[dtype][stack]

    def save_volume_to_tiff(
        self,
        filename: os.PathLike | str,
        index: int,
        dtype: Literal["u1", "f4", "u2"] = "u2",
    ):
        stack = self.get_volume(index, dtype=dtype, imagej=True)
        z_scale, y_scale, x_scale = self.header.scales
        tifffile.imwrite(
            filename,
            stack,
            imagej=True,
            resolution=(1.0 / x_scale, 1.0 / y_scale),
            metadata={
                "spacing": z_scale,
                "unit": "um",
                "axes": "TZCYX",
            },
            photometric="minisblack",
        )

    def save_all_volumes_to_tiff(
        self,
        filename: os.PathLike | str,
        dtype: Literal["u1", "f4", "u2"] = "u2",
        chunksize: int = 10,
    ):
        T, C, Z, Y, X = self.header.shape
        vol0 = self.get_volume(0, dtype=dtype, imagej=True)

        def frames():
            for start in range(0, T, chunksize):
                end = min(start + chunksize - 1, T - 1)
                stacks = self.get_multi_volumes(start, end, dtype, imagej=True)
                stacks = np.expand_dims(stacks, 0)
                yield from stacks

        tifffile.imwrite(
            filename,
            frames(),
            imagej=True,
            resolution=(1.0 / self.header.x_scale, 1.0 / self.header.y_scale),
            metadata={
                "spacing": self.header.z_scale,
                "unit": "um",
                "axes": "TZCYX",
            },
            photometric="minisblack",
            shape=(T, Z, C, Y, X),
            dtype=vol0.dtype,
        )

    def save_all_volumes_to_hdf(
        self,
        filename: os.PathLike | str,
        dtype: Literal["u1", "f4", "u2"] = "u2",
        chunksize: int = 10,
    ):
        if dtype not in LUT_TABLE:
            dtype = "u2"

        T, C, Z, Y, X = self.header.shape

        with h5py.File(filename, "w") as f:
            dset = f.create_dataset(
                "data",
                shape=(T, Z, C, Y, X),
                dtype=dtype,
                chunks=(1, Z, C, Y, X),
                **hdf5plugin.Blosc(
                    cname="zstd",
                    clevel=5,
                    shuffle=hdf5plugin.Blosc.BITSHUFFLE,
                ),
            )
            for start in range(0, T, chunksize):
                end = min(start + chunksize - 1, T - 1)
                stacks = self.get_multi_volumes(start, end, dtype, imagej=True)
                dset[start : end + 1, :, :, :, :] = stacks
