"""Memory-mapped reader for ISOP ``.bin`` slice files, plain or zipped.

File layout (all values big-endian)::

    N records, each:
        slice header   8 bytes       Y, X (2 x i4)
        slice data     Y*X x u2      stored in (Y, X) order

Example
-------
>>> rs = ISOPRawSlice("sample.bin")     # or "sample.bin.zip"
>>> rs.shape                            # (N, Y, X)
>>> img = rs.frame(0)                   # native uint16 copy, (Y, X)
"""

from __future__ import annotations

__all__ = ["ISOPRawSlice", "resolve_bin"]

import os
import shutil
import struct
import tempfile
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Final

import numpy as np
from numpy.typing import NDArray


def resolve_bin(path: os.PathLike[str] | str) -> tuple[Path, str | None]:
    """Return (file, member): a plain .bin, or a zip and its single .bin entry."""
    p = Path(path)
    for cand in (p, p.with_name(p.name + ".zip")):
        if not cand.is_file():
            continue
        if not zipfile.is_zipfile(cand):
            return cand.resolve(), None
        with zipfile.ZipFile(cand) as zf:
            members = [
                n
                for n in zf.namelist()
                if n.endswith(".bin") and not n.startswith("__MACOSX")
            ]
        if len(members) != 1:
            raise ValueError(f"{cand} must hold exactly one .bin, found {members}")
        return cand.resolve(), members[0]
    raise FileNotFoundError(f"Could not find {p} or {p.name}.zip")


class ISOPRawSlice:
    """Zero-copy, memory-mapped view of a ``.bin`` or  ``.bin.zip`` file.

    The whole file is mapped once, read-only, with a structured dtype
    (slice header + slice data per record). Nothing is read from disk
    until the data is accessed.

    Parameters
    ----------
    filepath
        Path to a .bin file, or a .bin.zip containing one bin file.

    Attributes
    ----------
    filepath : Path
        Path to the file.
    shape : tuple of int
        ``(N, Y, X)``.
    data : ndarray, shape (N, Y, X), dtype >u2
        Zero-copy big-endian view of all volumes.

    Raises
    ------
    ValueError
        If each hdr cannot be parsed, or the file is shorter
        than the header says it should be.

    Notes
    -----
    There is no ``close()``. The mapping is released automatically once
    this object and every array taken from it have been garbage-collected,
    so returned arrays stay valid after the reader is gone.
    Instances are immutable after construction and safe to share between threads.
    They can also be passed to other processes; see __reduce__.
    """

    _SLICE_HEADER: Final = 8
    """Bytes of header before each volume (2 x big-endian uint32)."""

    filepath: Path
    shape: tuple[int, int, int]
    data: NDArray[np.uint16]

    def __init__(
        self,
        filepath: os.PathLike[str] | str,
    ) -> None:
        self.filepath, name = resolve_bin(filepath)

        if name is not None:
            with zipfile.ZipFile(self.filepath) as zf:
                with zf.open(name) as src, tempfile.TemporaryFile() as tmp:
                    shutil.copyfileobj(src, tmp, 16 << 20)
                    tmp.flush()
                    self._map(tmp)
        else:
            with self.filepath.open("rb") as f:
                self._map(f)

    def _map(self, f: BinaryIO) -> None:
        """Parse the header and map the slices from an open file; f may be closed afterwards."""
        f.seek(0)
        raw = f.read(self._SLICE_HEADER)
        try:
            Y, X = struct.unpack(">2i", raw)
        except struct.error as e:
            raise ValueError(f"Not a valid .bin file: {self.filepath}") from e

        if Y <= 0 or X <= 0:
            raise ValueError(f"Invalid slice size {Y}x{X} in header: {self.filepath}")

        record = np.dtype([("hdr", ">i4", (2,)), ("data", ">u2", (Y, X))])
        try:
            mm = np.memmap(f, dtype=record, mode="r")  # no shape: NumPy checks the size
        except ValueError as e:
            raise ValueError(
                f"File size is not a whole number of {Y}x{X} slices: {self.filepath}"
            ) from e
        if not (mm["hdr"] == (Y, X)).all():
            raise ValueError(f"Slices have different sizes: {self.filepath}")

        self.shape = (len(mm), Y, X)
        self.data = mm["data"]

    def __repr__(self) -> str:
        return f"ISOPRawSlice({str(self.filepath)!r}, shape={self.shape})"

    def __reduce__(self) -> tuple[type[ISOPRawSlice], tuple[str]]:
        # Pickle only the path; the receiving process re-parses the header and re-maps.
        return (self.__class__, (str(self.filepath),))

    def __len__(self) -> int:
        """Number of slices ``N``."""
        return self.shape[0]

    def __getitem__(self, key: Any) -> NDArray[np.uint16]:
        """Index :attr:`data` with any NumPy index; returns a zero-copy view.

        ``rs[i]`` gives ``(Y, X)``. Slices, negative indices and
        multi-axis indices such as ``rs[i, jj]`` or ``rs[2:5, :, 10]`` also
        work. The result is big-endian ``>u2``; use :meth:`volume` for a
        native-endian copy.
        """
        result: NDArray[np.uint16] = self.data[key]
        return result

    def volume(self, idx: int) -> NDArray[np.uint16]:
        """Return slice ``idx`` as a native-endian uint16 copy.

        Parameters
        ----------
        idx
            Slice index. Negative values count from the end.

        Returns
        -------
        ndarray, shape (Y, X), dtype uint16
            C-contiguous array in the machine's native byte order, ready
            for libraries that reject big-endian data (PyTorch, Numba, ...).
        """
        result: NDArray[np.uint16] = self.data[idx].astype(np.uint16)
        return result
