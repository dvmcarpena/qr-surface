from pathlib import Path
from pprint import pprint

from tfginfo.qr import Correction
from tfginfo.error import QRErrorId
from tfginfo.images import Deformation, LabeledImage, parse_labeled_images

images_dir = (Path(__file__).parent.parent / "images").resolve()


def filter_func(labeled_image: LabeledImage) -> bool:
    return (
        # labeled_image.original_id == "bluephage"
        # labeled_image.dataset == "colorsensing"
        # labeled_image.dataset == "colorsensing2"
        # labeled_image.dataset == "synthetic_small"
        labeled_image.dataset in ["synthetic_small", "colorsensing", "colorsensing2"]
        and labeled_image.localization_error is None
        # and all(
        #     qr.deformation == Deformation.PERSPECTIVE or qr.deformation == Deformation.AFFINE
        #     for qr in labeled_image.qrs
        # )
        # and labeled_image.version == 7
        # and labeled_image.correction_results["projective"] == "wrong_version".upper()
        # and labeled_image.correction_results["projective"] == "ERROR_FEATURES"
        # and labeled_image.correction_results["projective"] == "CANT_READ"
        # and labeled_image.correction_results["projective"] == "NOT_ENOUGH_QRS"
        # and labeled_image.correction_results["projective"] == "WRONG_PIXELS"
        # and (labeled_image.method == Deformation.PERSPECTIVE
        #      or labeled_image.method == Deformation.AFFINE)
        # and labeled_image.method == Deformation.AFFINE
        # and labeled_image.method == Deformation.CYLINDRIC
        # and labeled_image.num_qrs == 1
        # and labeled_image.has_error()
        # and not labeled_image.has_error()
        # and labeled_image.error_id == QRErrorId.CANT_READ
        # and labeled_image.error_id == QRErrorId.ERROR_FEATURES
    )

corrections = [
    Correction.AFFINE,
    Correction.PROJECTIVE,
    Correction.CYLINDRICAL,
    Correction.TPS,
    Correction.DOUBLE_TPS
]

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

RELATIVE = False
RELATIVE_PERCENT = False

target_images = parse_labeled_images(
    images_dir,
    filter_func=filter_func
)

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
                if qr.correction_error[correction] is None:
                    results[correction.name]["GOOD"] += 1
                else:
                    results[correction.name][qr.correction_error[correction].name] += 1
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


precision = 4
if RELATIVE:
    for key, item in localization.items():
        if RELATIVE_PERCENT:
            localization[key] = round(100 * item / total, precision)
        else:
            localization[key] = round(item / total, precision)
    for key1, item1 in results.items():
        for key2, item2 in results[key1].items():
            if RELATIVE_PERCENT:
                item1[key2] = round(100 * item2 / total, precision)
            else:
                item1[key2] = round(item2 / total, precision)

pprint(localization)
pprint(results)
pprint(zbar)
