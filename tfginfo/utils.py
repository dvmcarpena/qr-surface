import copy
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from tfginfo.datasets import Deformation, BitmapCollection, LabeledImage
from tfginfo.qrsurface import BadModules, Correction, decode, Features, QRCode, QRErrorId


def try_decode_with_zbar(labeled_image: LabeledImage, image: np.ndarray) -> Optional[QRErrorId]:
    if not labeled_image.has_data:
        return None

    try:
        results = decode.decode(image)
    except ValueError:
        results = []

    if labeled_image.num_qrs != len(results):
        zbar_err = QRErrorId.NOT_ENOUGH_QRS
    elif any(labeled_image.data != message for message in results):
        zbar_err = QRErrorId.BAD_DATA
    else:
        zbar_err = None

    return zbar_err


def parse_qrs(image: np.ndarray, features: Features) -> List[QRCode]:
    try:
        return list(QRCode.from_features(image, features))
    except Exception:
        features.plot()
        raise QRErrorId.ERROR_FEATURES.exception()


def check_num_qrs(labeled_image: LabeledImage, features: Features, qrs: List[QRCode]):
    if len(qrs) < labeled_image.num_qrs:
        features.plot()
        raise QRErrorId.NOT_ENOUGH_QRS.exception()
    elif len(qrs) == 0:
        features.plot()


def check_version(labeled_image: LabeledImage, features: Features, qrs: List[QRCode]):
    for i, qr in enumerate(qrs):
        if labeled_image.version is not None:
            if labeled_image.qrs[i].deformation == Deformation.SURFACE:
                qr.update_version(labeled_image.version)
            elif labeled_image.version != qr.version:
                features.plot()
                raise QRErrorId.WRONG_VERSION.exception(
                    estimated=qr.version,
                    expected=labeled_image.version
                )


def sort_qrs(qrs: List[QRCode]) -> List[QRCode]:
    if len(qrs) == 0:
        return qrs

    points = np.array([qr.finder_patterns[0].center for qr in qrs])
    return list(np.array(qrs)[np.lexsort((points[:, 0], points[:, 1]))])


def check_correction(labeled_image: LabeledImage, features: Features, qr: QRCode,
                     bitmaps: BitmapCollection, correction: Correction) -> Optional[BadModules]:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle(f"{labeled_image.path.name} {correction.name}")

    features.plot(axes=ax1)
    qr.plot(axes=ax2)

    correct_qr = copy.deepcopy(qr)
    correct_qr.correct(method=correction, bitpixel=5, border=8, simple=True)
    correct_qr.plot(axes=ax3)

    if not labeled_image.has_data:
        return None

    bitmap = bitmaps.get_bitmap(labeled_image)
    sampled_qr = qr.sample(method=correction)

    sampled_qr.plot_differences(bitmap, axes=ax4)

    errors = sampled_qr.count_errors(bitmap)
    bad_modules = BadModules(
        count=errors,
        relative=errors / (qr.size**2)
    )

    try:
        data = sampled_qr.decode()
    except ValueError:
        raise QRErrorId.CANT_READ.correction_exception(bad_modules)

    if data != labeled_image.data:
        print(data)
        print(labeled_image.data)
        raise QRErrorId.BAD_DATA.correction_exception(bad_modules)

    if errors > 0:
        raise QRErrorId.WRONG_PIXELS.correction_exception(bad_modules, num_pixels=errors)

    return bad_modules
