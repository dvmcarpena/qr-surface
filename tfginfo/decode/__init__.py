from collections import namedtuple
from enum import auto, Enum, unique
from typing import Callable, Dict, List, Optional

import numpy as np
from pyzbar import pyzbar as zbar

from tfginfo.utils import Image

ZBarResult = namedtuple('ZBarResult', ['data', 'type', 'rect', 'polygon'])


def image_to_ocvimage(image: Image) -> Image:
    return image[:, :, ::-1]


def decode_zbar(img: Image) -> List[str]:
    contents: List[ZBarResult] = zbar.decode(np.array(img))

    if len(contents) == 0:
        raise ValueError("No readable QR found")

    return list(map(lambda x: x.data.decode("utf-8"), contents))


# import cv2
# def decode_opencv(image: Image, epsx: Optional[float] = None,
#                   epsy: Optional[float] = None) -> List[str]:
#     image = image_to_ocvimage(image)
#
#     detector = cv2.QRCodeDetector()
#     if epsx is not None:
#         detector.setEpsX(epsx)
#     if epsy is not None:
#         detector.setEpsY(epsy)
#
#     # TODO check if detects multiple QRs
#     data, *_ = detector.detectAndDecode(image)
#     return [data]


@unique
class DecodeMethods(Enum):
    ZBAR = auto()
    # OPENCV = auto()
    # ZXING = auto()


_Decoder = Callable[..., List[str]]
_DECODE_METHOD_FUNCS: Dict[DecodeMethods, _Decoder] = {
    DecodeMethods.ZBAR: decode_zbar,
    # DecodeMethods.OPENCV: decode_opencv
}


def decode(image: Image, method: Optional[DecodeMethods] = None, **kwargs) -> List[str]:
    if method is None:
        method = DecodeMethods.ZBAR

    func = _DECODE_METHOD_FUNCS[method]
    return func(image, **kwargs)
