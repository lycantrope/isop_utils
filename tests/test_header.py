import struct
from pathlib import Path

import numpy as np
import pytest

from isop_utils import ISOPHeader, ISOPVirtualStack

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
    with ISOPVirtualStack(file) as stack:
        v1 = stack.get_volume(0)
        v2 = stack.get_volume(1)
        np.testing.assert_equal(v2 - v1, np.ones_like(v1))


def testing_not_exist(file: Path):
    data_path = file.with_name("not_exists.3du16")
    with pytest.raises(FileNotFoundError):
        print(ISOPHeader.from_3DU16(data_path))


def test_readfile(file: Path):
    test_out = file.parent.joinpath("tmp")
    test_out.mkdir(exist_ok=True)
    with ISOPVirtualStack(file) as stack, pytest.warns(UserWarning):
        for i in range(4):
            for fmt in ("org", "u1", "f4"):
                out = test_out.joinpath(file.stem + f"_t={i:0>5d}_{fmt}.tif")
                stack.save_volume_to_tiff(out, i, dtype=fmt)  # pyright: ignore[reportArgumentType]


def test_read_volume_fail(file: Path):
    test_out = file.parent.joinpath("tmp")
    test_out.mkdir(exist_ok=True)
    with pytest.raises(IndexError):
        with ISOPVirtualStack(file) as stack:
            stack.get_volume(100)


def test_read_volume_as_imagej(file: Path):
    with ISOPVirtualStack(file) as stack:
        # This method should return 1 volume of image stack with format (1, Z, C, Y, X)
        img = stack.get_volume(3, imagej=True)
        assert img.ndim == 5
        assert img.shape == (1, Z, C, Y, X)


def test_save_all_volume(file: Path):
    test_out = file.parent.joinpath("tmp")
    test_out.mkdir(exist_ok=True)
    out = test_out.joinpath(file.stem + ".tif")
    with ISOPVirtualStack(file) as stack:
        stack.save_all_volumes_to_tiff(out)


def test_save_all_volume_with_small_chunk(file: Path):
    test_out = file.parent.joinpath("tmp")
    test_out.mkdir(exist_ok=True)
    out = test_out.joinpath(file.stem + ".tif")
    with ISOPVirtualStack(file) as stack:
        stack.save_all_volumes_to_tiff(out, chunksize=1)


def test_read_volumes(file: Path):
    test_out = file.parent.joinpath("tmp")
    test_out.mkdir(exist_ok=True)
    with ISOPVirtualStack(file) as stack:
        T, C, Z, Y, X = stack.header.shape
        imgs = stack.get_multi_volumes(0, 10)
        assert imgs.shape == (11, C, Z, Y, X)
        imgs = stack.get_multi_volumes(0, 10, imagej=True)
        assert imgs.shape == (11, Z, C, Y, X)


def test_read_volume_out_of_bound(file: Path):
    test_out = file.parent.joinpath("tmp")
    test_out.mkdir(exist_ok=True)
    with ISOPVirtualStack(file) as stack, pytest.raises(IndexError):
        T, C, Z, Y, X = stack.header.shape
        stack.get_volume(T + 100)


def test_read_multi_volumes_out_of_bound(file: Path):
    test_out = file.parent.joinpath("tmp")
    test_out.mkdir(exist_ok=True)
    with ISOPVirtualStack(file) as stack, pytest.raises(IndexError):
        stack.get_multi_volumes(stack.header.n_frame + 100, 0)


def test_save_all_volume_via_hdf(file: Path):
    test_out = file.parent.joinpath("tmp")
    test_out.mkdir(exist_ok=True)
    out = test_out.joinpath(file.stem + ".h5")
    with ISOPVirtualStack(file) as stack:
        stack.save_all_volumes_to_hdf(out)
