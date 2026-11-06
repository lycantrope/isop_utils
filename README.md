# ISOP Utils

`scape_utils` is a Python module designed for handling `.3DU16` files, providing tools for reading and manipulating volumetric data. The module includes two main classes: `ScapeVirtualStack` and `ScapeImageDecoder`.

## Installation

To install `scape_utils`, use pip:

```bash
pip install git+https://github.com/lycantrope/scape_util
```

## Usage

To read a `.3DU16` file, use the `ScapeVirtualStack` class. Below are examples of how to get volumes and ImageJ formatted volumes.

```python
from pathlib import Path

import numpy as np
from isop_utils import ISOPVirtualStack

file_path = Path("path/to/your/file.3DU16")

with ISOPVirtualStack(file_path) as stack:
    # show information
    print(stack)

    # Get the 1st volume in format (TCZYX)
    volume_0 = stack.get_volume(0)
    
    # Get the 3rd volume in ImageJ format (TZCYX)
    volume_ij = stack.get_volume(2, imagej=True)

    # Get multiple stacks at once (from t0 to t10)
    volume_multi = stack.get_multi_volumes(0, 10)

    # save all volumes into ImageJ compatible tiff
    stack.save_all_volumes_to_tiff(file_path.with_suffix(".tif"))

```

## License

This project is licensed under the MIT License.
