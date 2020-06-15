from enum import auto, Enum, unique
from typing import Callable, Dict, List, Optional

import numpy as np

from .zbar import decode_zbar


@unique
class DecodeMethods(Enum):
    ZBAR = auto()
    # OPENCV = auto()


_Decoder = Callable[..., List[str]]
_DECODE_METHOD_FUNCS: Dict[DecodeMethods, _Decoder] = {
    DecodeMethods.ZBAR: decode_zbar,
    # DecodeMethods.OPENCV: decode_opencv
}


def decode(image: np.ndarray, method: Optional[DecodeMethods] = None, **kwargs) -> List[str]:
    if method is None:
        method = DecodeMethods.ZBAR

    func = _DECODE_METHOD_FUNCS[method]
    return func(image, **kwargs)
