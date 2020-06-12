# import cv2
import numpy as np


def image_to_ocvimage(image: np.ndarray) -> np.ndarray:
    return image[:, :, ::-1]


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
