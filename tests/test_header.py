import struct
from pathlib import Path

import numpy as np
import pytest

from isop_utils import ISOPVirtualStack, isop_3du16_to_tiff

T, C, Z, Y, X = 31, 2, 3, 5, 7


@pytest.fixture
def file():
    sample_path = Path(__file__).parent.joinpath("sample.3DU16")
    if sample_path.is_file():
        return sample_path

    # 3, z_scale, y_scale, x_scale
    z_scale, y_scale, x_scale = 0.9, 0.455, 0.455
    scale = (3, z_scale, y_scale, x_scale)

    # 5, T, C, Z, Y, X
    metadata = (5, T, C, Z, Y, X)
    # 1 volume of data
    # C, Z, Y, Z, [data...]
    data = [C, Z, Y, X, *range(C * Z * Y * X)]

    raw = struct.pack(
        ">i3d6i" + f"4i{C * Z * Y * X:d}H" * T,
        *scale,
        *metadata,
        *[d + i for i in range(T) for d in data],
    )

    with open(sample_path, "wb") as f:
        f.write(raw)

    return sample_path


def testing_parser(file: Path):
    stack = ISOPVirtualStack(file)
    v1 = stack[0]
    v2 = stack[1]
    np.testing.assert_equal(v2 - v1, np.ones_like(v1))


def test_read_raw(file: Path):
    test_out = file.parent.joinpath("tmp")
    test_out.mkdir(exist_ok=True)
    stack = ISOPVirtualStack(file)
    imgs = stack[0:11]
    assert imgs.shape == (11, *stack.shape[1:])
    assert imgs.dtype == np.dtype(">u2")


def test_read_raw_fail(file: Path):
    test_out = file.parent.joinpath("tmp")
    test_out.mkdir(exist_ok=True)
    stack = ISOPVirtualStack(file)
    with pytest.raises(IndexError):
        _ = stack[100]


def test_read_volume(file: Path):
    test_out = file.parent.joinpath("tmp")
    test_out.mkdir(exist_ok=True)
    stack = ISOPVirtualStack(file)
    im = stack.volume(5)

    assert im.shape == stack.shape[1:]
    assert im.dtype == np.dtype("<u2")


def test_read_volume_fail(file: Path):
    test_out = file.parent.joinpath("tmp")
    test_out.mkdir(exist_ok=True)
    stack = ISOPVirtualStack(file)
    with pytest.raises(IndexError):
        _ = stack.volume(100)


def test_save_all_volume(file: Path):
    test_out = file.parent.joinpath("tmp")
    test_out.mkdir(exist_ok=True)
    out = test_out.joinpath(file.stem + ".tif")
    stack = ISOPVirtualStack(file)
    isop_3du16_to_tiff(stack, out)
