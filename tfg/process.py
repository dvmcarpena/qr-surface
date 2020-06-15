import copy
from pathlib import Path
from pprint import pprint
from typing import List

import matplotlib.pyplot as plt

from tfg.qrsurface import Correction, CorrectionException, Features, QRException
from tfg.datasets import LabeledImage, BitmapCollection
from tfg.utils import check_correction, check_num_qrs, check_version, parse_qrs, sort_qrs, try_decode_with_zbar


def process_results(target_images: List[LabeledImage], corrections: List[Correction], images_dir: Path,
                    update: bool = False, show_diff: bool = True, plot_all: bool = False, plot_success: bool = False,
                    plot_localization_errors: bool = False, plot_correction_errors: bool = False) -> None:
    bitmaps = BitmapCollection(images_dir)

    if show_diff:
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

    total = len(target_images)
    for i, labeled_image in enumerate(target_images):
        short_path = labeled_image.path.relative_to(images_dir)
        print(f"{i + 1:02d}/{total} - {short_path}")

        image = labeled_image.read_image()
        updated = False
        try:
            zbar_err = try_decode_with_zbar(labeled_image, image)

            updated = labeled_image.update_zbar(labeled_image.has_data, zbar_err, diff, update=update)

            features = Features.from_image(image)
            qrs = parse_qrs(image, features)
            check_num_qrs(labeled_image, features, qrs)
            qrs = sort_qrs(qrs)
            check_version(labeled_image, features, qrs)

            for correction in corrections:
                cqrs = copy.deepcopy(qrs)

                for j, qr in enumerate(cqrs):
                    try:
                        bad_modules = check_correction(labeled_image, features, qr, bitmaps, correction)

                        if plot_all or plot_success:
                            plt.show()
                        else:
                            plt.close()
                        updated = labeled_image.update_successful_correction(correction, j, bad_modules, diff,
                                                                             update=update)
                    except CorrectionException as e:
                        if plot_all or plot_correction_errors:
                            plt.show()
                        else:
                            plt.close()
                        updated = labeled_image.update_correction_error(correction, j, e.bad_modules, e.error, diff,
                                                                        update=update)

                del cqrs

        except QRException as e:
            if plot_all or plot_localization_errors:
                plt.show()
            else:
                plt.close()
            updated = labeled_image.update_localization_error(e.error, diff, update=update)
        except Exception as e:
            raise e

        if updated:
            print()

    if show_diff:
        pprint(diff)
