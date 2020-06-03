import copy
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from tfginfo.features import Features
from tfginfo.qr import QRCode, Correction, MatchingFeatures
from tfginfo.error import QRErrorId
from tfginfo.images import BadModules, Deformation, get_original_qr, LabeledImage
from tfginfo.utils import Image
from tfginfo.utils import get_size_from_version


def parse_qrs(image: Image, features: Features) -> List[QRCode]:
    try:
        return list(QRCode.from_features(image, features))
    except Exception:
        # import traceback
        # print(traceback.format_exc())
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
                qr.version = labeled_image.version
                qr.size = get_size_from_version(qr.version)
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
                     original_qrs, correction: Correction) -> Optional[BadModules]:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle(f"{labeled_image.path.name} {correction.name}")

    features.plot(axes=ax1)
    qr.plot(axes=ax2)

    # if len(list(filter(lambda a: a is not None, qr.alignment_patterns))) == 0 and correction == Correction.PROJECTIVE:
    #     qr.correct(method=correction, bitpixel=5, border=10, references_features=[
    #         MatchingFeatures.FINDER_CENTERS,
    #         MatchingFeatures.FINDER_CORNERS,
    #         MatchingFeatures.ALIGNMENTS_CENTERS,
    #         MatchingFeatures.FOURTH_CORNER
    #     ])
    # else:
    correct_qr = copy.deepcopy(qr)
    correct_qr.correct(method=correction, bitpixel=5, border=8)
    # correct_qr.correct(method=correction, bitpixel=5, border=0, simple=True)
    correct_qr.plot(axes=ax3)

    if not labeled_image.has_data:
        return None

    original_qr = get_original_qr(original_qrs, labeled_image)
    sampled_qr = qr.sample(method=correction)

    sampled_qr.plot_differences(original_qr, axes=ax4)

    errors = sampled_qr.count_errors(original_qr)
    bad_modules = BadModules(
        count=errors,
        relative=errors / (get_size_from_version(qr.version)**2)
    )

    try:
        # data = qr.decode(bounding_box=False, sample=True)
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

