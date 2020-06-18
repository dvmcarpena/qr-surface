from pathlib import Path

import click

from tfg import process_results, read_results

images_dir = (Path(__file__).parent.parent / "data" / "datasets").resolve()


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
    read_results(images_dir)


cli()
