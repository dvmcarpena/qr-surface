from collections import namedtuple
from typing import List

import numpy as np
from pyzbar import pyzbar as zbar


ZBarResult = namedtuple('ZBarResult', ['data', 'type', 'rect', 'polygon'])


def decode_zbar(image: np.ndarray) -> List[str]:
    contents: List[ZBarResult] = zbar.decode(np.array(image))

    if len(contents) == 0:
        raise ValueError("No readable QR found")

    return list(map(lambda x: x.data.decode("utf-8"), contents))
