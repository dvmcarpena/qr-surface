from pprint import pprint
from typing import List

from tfginfo.qrsurface import Correction, QRErrorId
from tfginfo.datasets import LabeledImage


def read_results(target_images: List[LabeledImage], corrections: List[Correction], relative: bool = False,
                 relative_percent: bool = True, precision: int = 4) -> None:
    localization_errors = [
        QRErrorId.ERROR_FEATURES,
        QRErrorId.NOT_ENOUGH_QRS,
        QRErrorId.WRONG_VERSION
    ]
    correction_errors = [
        QRErrorId.CANT_READ,
        QRErrorId.BAD_DATA,
        QRErrorId.WRONG_PIXELS
    ]

    localization = {
        err_id: 0
        for err_id in map(lambda e: e.name, localization_errors)
    }
    results = {
        correction: {
            err_id: 0
            for err_id in map(lambda e: e.name, correction_errors)
        }
        for correction in map(lambda e: e.name, corrections)
    }

    for correction in map(lambda e: e.name, corrections):
        results[correction]["GOOD"] = 0

    total = 0
    for labeled_image in target_images:
        if labeled_image.localization_error is None:
            total += labeled_image.num_qrs
        else:
            total += 1

    print(total)
    for labeled_image in target_images:
        if labeled_image.localization_error is None:
            for i, qr in enumerate(labeled_image.qrs):
                for correction in qr.correction_error.keys():
                    try:
                        if qr.correction_error[correction] is None:
                            results[correction.name]["GOOD"] += 1
                        else:
                            results[correction.name][qr.correction_error[correction].name] += 1
                    except KeyError:
                        pass
        else:
            localization[labeled_image.localization_error.name] += 1

    zbar = {
        "GOOD": 0,
        QRErrorId.BAD_DATA.name: 0,
        QRErrorId.NOT_ENOUGH_QRS.name: 0
    }
    for labeled_image in target_images:
        if labeled_image.zbar:
            if labeled_image.zbar_error:
                zbar[labeled_image.zbar_error.name] += 1
            else:
                zbar["GOOD"] += 1

    if relative:
        for key, item in localization.items():
            if relative_percent:
                localization[key] = round(100 * item / total, precision)
            else:
                localization[key] = round(item / total, precision)
        for key1, item1 in results.items():
            for key2, item2 in results[key1].items():
                if relative_percent:
                    item1[key2] = round(100 * item2 / total, precision)
                else:
                    item1[key2] = round(item2 / total, precision)

    pprint(localization)
    pprint(results)
    pprint(zbar)
