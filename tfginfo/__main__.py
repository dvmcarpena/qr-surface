import copy
from pathlib import Path
from pprint import pprint
from typing import Optional
import warnings

import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

from tfginfo.utils import Image
from tfginfo.features import Features
from tfginfo.decode import decode
from tfginfo.qr import Correction
from tfginfo.error import QRErrorId, QRException, CorrectionException
from tfginfo.images import LabeledImage, parse_labeled_images, parse_original_qrs, Deformation
from tfginfo.validation import check_correction, check_num_qrs, check_version, parse_qrs, sort_qrs

warnings.filterwarnings('error')

PLOT_ALL = False
PLOT_SUCCESS = False
PLOT_CORRECTION_ERRORS = False
PLOT_LOCALIZATION_ERRORS = False
UPDATE_DATASET = False
DIFF = True

if __name__ == "__main__":
    images_dir = (Path(__file__).parent.parent / "images").resolve()

    def filter_func(labeled_image: LabeledImage) -> bool:
        return (
            # labeled_image.dataset == "bluephage"
            # labeled_image.dataset == "colorsensing"
            # labeled_image.dataset == "colorsensing2"
            # labeled_image.dataset == "synthetic_small"
            labeled_image.dataset in ["synthetic_small", "colorsensing", "colorsensing2"]
            # and labeled_image.version == 7
            # and labeled_image.num_qrs > 1
            and labeled_image.localization_error is None
            # and all(
            #     qr.deformation == Deformation.CYLINDRIC
            #     for qr in labeled_image.qrs
            # )
            # and any(
            #     items == QRErrorId.WRONG_PIXELS
            #     for qr in labeled_image.qrs
            #     if qr.correction_error is not None
            #     for items in qr.correction_error.values()
            # )
            # and all(
            #     qr.deformation == Deformation.PERSPECTIVE or qr.deformation == Deformation.AFFINE
            #     for qr in labeled_image.qrs
            # )
            # and labeled_image.method == Deformation.AFFINE
            # and labeled_image.method == Deformation.CYLINDRIC
            # and labeled_image.num_qrs == 1
            # and labeled_image.has_error()
            # and not labeled_image.has_error()
            # and labeled_image.error_id == QRErrorId.CANT_READ
            # and labeled_image.error_id == QRErrorId.ERROR_FEATURES
        )

    original_qrs = parse_original_qrs(images_dir)
    target_images = parse_labeled_images(
        images_dir,
        filter_func=filter_func
    )

    corrections = [
        Correction.AFFINE,
        Correction.PROJECTIVE,
        Correction.CYLINDRICAL,
        Correction.TPS,
        # Correction.DOUBLE_TPS
    ]
    if DIFF:
        diff = {
            "localization_error": {},
            "correction_error": {
                correction.name: {}
                for correction in corrections
            },
            "bad_modules": {
                correction.name: {}
                for correction in corrections
            },
            "zbar_err": {}
        }
    else:
        diff = None

    # with tqdm(total=len(target_images), ncols=150) as progress_bar:
    # with tqdm(total=len(target_images)) as progress_bar:
    total = len(target_images)
    for i, labeled_image in enumerate(target_images):
        short_path = labeled_image.path.relative_to(images_dir)
        # progress_bar.set_description(f"{short_path}\n", refresh=False)
        print(f"{i + 1:02d}/{total} - {short_path}")

        image: Image = imageio.imread(str(labeled_image.path))

        if UPDATE_DATASET:
            labeled_image.update_legacy()
        updated = False
        try:
            zbar_err = None
            if labeled_image.has_data:
                try:
                    results = decode(image)
                except ValueError:
                    results = []

                if labeled_image.num_qrs != len(results):
                    zbar_err = QRErrorId.NOT_ENOUGH_QRS
                elif any(labeled_image.data != message for message in results):
                    zbar_err = QRErrorId.BAD_DATA
                else:
                    zbar_err = None

            updated = labeled_image.update_zbar(labeled_image.has_data, zbar_err, diff, update=UPDATE_DATASET)

            features = Features.from_image(image)
            qrs = parse_qrs(image, features)
            check_num_qrs(labeled_image, features, qrs)
            qrs = sort_qrs(qrs)
            check_version(labeled_image, features, qrs)

            correction_results = {}
            for correction in corrections:
                cqrs = copy.deepcopy(qrs)

                for j, qr in enumerate(cqrs):
                    try:
                        bad_modules = check_correction(labeled_image, features, qr, original_qrs, correction)

                        if PLOT_ALL or PLOT_SUCCESS:
                            plt.show()
                        else:
                            plt.close()
                        updated = labeled_image.update_successfull_correction(correction, j, bad_modules, diff,
                                                                              update=UPDATE_DATASET)
                    except CorrectionException as e:
                        if PLOT_ALL or PLOT_CORRECTION_ERRORS:
                            plt.show()
                        else:
                            plt.close()
                        updated = labeled_image.update_correction_error(correction, j, e.bad_modules, e.error, diff,
                                                                        update=UPDATE_DATASET)

                del cqrs

        except QRException as e:
            if PLOT_ALL or PLOT_LOCALIZATION_ERRORS:
                plt.show()
            else:
                plt.close()
            updated = labeled_image.update_localization_error(e.error, diff, update=UPDATE_DATASET)
        except Exception as e:
            # print()
            # print("INTERNAL ERROR")
            raise e

        if updated:
            print()

        # progress_bar.update()

    # pprint(results_old)
    # pprint(results)
    if DIFF:
        pprint(diff)
