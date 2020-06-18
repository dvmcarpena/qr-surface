import copy
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt

from tfg.qrsurface import Correction, CorrectionException, Features, QRException
from tfg.datasets import LabeledImage, BitmapCollection
from tfg.utils import check_correction, check_num_qrs, check_version, parse_qrs, sort_qrs, try_decode_with_zbar


def process_results(images_dir: Path, update: bool = False, show_diff: bool = True, plot_all: bool = False,
                    plot_success: bool = False, plot_localization_errors: bool = False,
                    plot_correction_errors: bool = False) -> None:
    """
    Searches for datasets in the path given and produces results for all the images in them

    :param images_dir: Root path for searching for datasets
    :param update: Whether to update the saved results
    :param show_diff: Show the differences between this execution and the last saved
    :param plot_all: Whether to plot all images
    :param plot_success: Whether to plot images without errors
    :param plot_localization_errors:  Whether to plot images with localization errors
    :param plot_correction_errors: Whether to plot images with correction errors
    """

    def filter_func(labeled_image: LabeledImage) -> bool:
        return (
            # labeled_image.dataset == "flat"
            labeled_image.dataset == "random"
            # labeled_image.datasety == "synthetic_small"
            # labeled_image.dataset in ["flat", "random", "synthetic_small"]
            # and labeled_image.localization_error is None
            # and labeled_image.has_error()
            # and not labeled_image.has_error()
            # and labeled_image.version == 7
            # and labeled_image.num_qrs > 1
            # and labeled_image.num_qrs == 1
            # and all(
            #     qr.deformation == Deformation.PERSPECTIVE
            #     for qr in labeled_image.qrs
            # )
            # and all(
            #     qr.deformation == Deformation.AFFINE
            #     for qr in labeled_image.qrs
            # )
            # and all(
            #     qr.deformation == Deformation.PERSPECTIVE or qr.deformation == Deformation.AFFINE
            #     for qr in labeled_image.qrs
            # )
            # and all(
            #     qr.deformation == Deformation.CYLINDRIC
            #     for qr in labeled_image.qrs
            # )
            # and all(
            #     qr.deformation == Deformation.SURFACE
            #     for qr in labeled_image.qrs
            # )
            # and any(
            #     items == QRErrorId.WRONG_PIXELS
            #     for qr in labeled_image.qrs
            #     if qr.correction_error is not None
            #     for items in qr.correction_error.values()
            # )
        )

    corrections = [
        Correction.AFFINE,
        Correction.PROJECTIVE,
        Correction.CYLINDRICAL,
        Correction.TPS,
        # Correction.DOUBLE_TPS
    ]

    target_images = LabeledImage.search_labeled_images(images_dir, filter_func=filter_func)

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
