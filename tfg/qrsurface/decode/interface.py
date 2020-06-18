from enum import auto, Enum, unique
from typing import Callable, Dict, List, Optional

import numpy as np

from .zbar import decode_zbar


@unique
class DecodeMethods(Enum):
    """
    Identifiers of the decoding methods
    """
    ZBAR = auto()
    # OPENCV = auto()


# Mapper from decoding method identifiers to its functions
_Decoder = Callable[..., List[str]]
_DECODE_METHOD_FUNCS: Dict[DecodeMethods, _Decoder] = {
    DecodeMethods.ZBAR: decode_zbar,
    # DecodeMethods.OPENCV: decode_opencv
}


def decode(image: np.ndarray, method: Optional[DecodeMethods] = None, **kwargs) -> List[str]:
    """
    Decodes a image with QR Codes

    :param image: Image
    :param method: Identifier of the decoding method
    :param kwargs: Keyword arguments for the decoding method

    :return: The list of data of each QR Code in the image
    """
    if method is None:
        method = DecodeMethods.ZBAR

    func = _DECODE_METHOD_FUNCS[method]
    return func(image, **kwargs)
