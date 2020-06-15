from pathlib import Path
import warnings

import click

from tfg import Correction, Deformation, LabeledImage, process_results, QRErrorId, read_results


def filter_func(labeled_image: LabeledImage) -> bool:
    return (
        # labeled_image.dataset == "colorsensing"
        # labeled_image.dataset == "colorsensing2"
        # labeled_image.dataset == "synthetic_small"
        labeled_image.dataset in ["synthetic_small", "colorsensing", "colorsensing2"]
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

warnings.filterwarnings('error')
images_dir = (Path(__file__).parent.parent / "data" / "datasets").resolve()
target_images = LabeledImage.search_labeled_images(images_dir, filter_func=filter_func)#[1:]


@click.group()
def cli():
    pass


@cli.command()
@click.option("--update", default=False, is_flag=True,
              help="Update the labels from the datasets with the results")
@click.option("--no-show-diff", default=True, is_flag=True,
              help="Show the difference within the current result and the saved ones")
@click.option("--plot-all", default=False, is_flag=True,
              help="Plot for all the target images")
@click.option("--plot-success", default=False, is_flag=True,
              help="Plot only the successful target images")
@click.option("--plot-localization-errors", default=False, is_flag=True,
              help="Plot only the target images with errors in the localization")
@click.option("--plot-correction-errors", default=False, is_flag=True,
              help="Plot only the target images with errors in the correction")
def process(update, no_show_diff, plot_all, plot_success, plot_localization_errors, plot_correction_errors):
    if update and not click.confirm("Are you sure that you want to update the saved results?"):
        return

    process_results(
        target_images=target_images,
        corrections=corrections,
        images_dir=images_dir,
        update=update,
        show_diff=no_show_diff,
        plot_all=plot_all,
        plot_success=plot_success,
        plot_localization_errors=plot_localization_errors,
        plot_correction_errors=plot_correction_errors
    )


@cli.command()
def results():
    read_results(target_images=target_images, corrections=corrections)


cli()
