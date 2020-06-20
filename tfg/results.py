import copy
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from matplotlib.figure import Figure, Axes

from tfg.qrsurface import Correction, QRErrorId
from tfg.datasets import LabeledImage, Deformation

IMAGE_ID = "image_id"
LOCALIZATION = "loc"
LOC_ERROR = "loc_error"
DATASET = "dataset"
DATASET_FLAT = "flat"
DATASET_RAND = "random"
DATASET_SYNT = "synthetic_small"
ZBAR = "zbar"
DEFORMATION = "deformation"
READ_AFF = "read_affine"
READ_PRO = "read_projective"
READ_CYL = "read_cylindrical"
READ_TPS = "read_tps"
RESULTS_DIR = Path("data/results")


def read_results(images_dir: Path) -> None:
    """
    Given a root path with datasets, reads all the last execution results and produces some meaningful output

    :param images_dir: Root path for searching for datasets
    """
    target_images = LabeledImage.search_labeled_images(images_dir)
    corrections = [
        Correction.AFFINE,
        Correction.PROJECTIVE,
        Correction.CYLINDRICAL,
        Correction.TPS,
        # Correction.DOUBLE_TPS
    ]

    # Building the DataFrame with all the data =========================================================================

    d = {
        IMAGE_ID: [
            labeled_image.image_id
            for labeled_image in target_images
            for _ in range(labeled_image.num_qrs)
        ],
        DATASET: [
            labeled_image.dataset
            for labeled_image in target_images
            for _ in range(labeled_image.num_qrs)
        ],
        LOCALIZATION: [
            labeled_image.localization_error is None
            for labeled_image in target_images
            for _ in range(labeled_image.num_qrs)
        ],
        LOC_ERROR: [
            labeled_image.localization_error.name if labeled_image.localization_error is not None else None
            for labeled_image in target_images
            for _ in range(labeled_image.num_qrs)
        ],
        ZBAR: [
            labeled_image.zbar and labeled_image.zbar_error is None
            for labeled_image in target_images
            for _ in range(labeled_image.num_qrs)
        ],
        DEFORMATION: [
            qr.deformation if qr is not None else None
            for labeled_image in target_images
            for qr in (labeled_image.qrs if labeled_image.num_qrs != 0 else [None])
        ]
    }

    read = {
        correction: []
        for correction in map(lambda e: e.name, corrections)
    }
    perfect = copy.deepcopy(read)
    num_errors = copy.deepcopy(read)
    rel_errors = copy.deepcopy(read)
    for labeled_image in target_images:
        if labeled_image.localization_error is None and labeled_image.num_qrs != 0:
            assert len(labeled_image.qrs) == labeled_image.num_qrs

            for i, qr in enumerate(labeled_image.qrs):
                assert len(qr.correction_error.keys()) == 5 or len(qr.correction_error.keys()) == 4
                for correction in qr.correction_error.keys():
                    try:
                        read[correction.name].append(qr.correction_error[correction] is not QRErrorId.CANT_READ)
                        perfect[correction.name].append(qr.correction_error[correction] is None)
                        num_errors[correction.name].append(qr.bad_modules[correction].count)
                        rel_errors[correction.name].append(qr.bad_modules[correction].relative)
                    except KeyError:
                        pass
        else:
            for _ in range(labeled_image.num_qrs):
                for correction in corrections:
                    read[correction.name].append(None)
                    perfect[correction.name].append(None)
                    num_errors[correction.name].append(None)
                    rel_errors[correction.name].append(None)

    for correction in corrections:
        id = correction.name.lower()
        d[f"read_{id}"] = read[correction.name]
        d[f"perfect_{id}"] = perfect[correction.name]
        d[f"num_errors_{id}"] = num_errors[correction.name]
        d[f"rel_errors_{id}"] = rel_errors[correction.name]

    corrections_labels = [correction.name[:3] for correction in corrections]
    df = pd.DataFrame(d)
    # print(df)

    # Analysing the results and producing outputs ======================================================================

    df_loc = df[df[LOCALIZATION]]
    total = df[IMAGE_ID].count()
    total_images = df[IMAGE_ID].drop_duplicates().count()
    total_loc = df[LOCALIZATION].sum()

    # Parameters of the experiments ------------------------------------------------------------------------------------
    print("Parameters of the experiments:")
    print(f" - Number of images:\t\t\t\t\t\t\t{total_images}")
    print(f" - Number of qrs:\t\t\t\t\t\t\t\t{total}")

    def count_images_and_qrs(name: str, dataset: str) -> None:
        num_images = df[df[DATASET] == dataset][IMAGE_ID].drop_duplicates().count()
        num_qrs = df[df[DATASET] == dataset][IMAGE_ID].count()
        print(f" - Number of images in the {name} dataset:\t\t{num_images}")
        print(f" - Number of qrs in the {name} dataset:\t\t\t{num_qrs}")
        del num_images
        del num_qrs

    count_images_and_qrs("flat", DATASET_FLAT)
    count_images_and_qrs("rand", DATASET_RAND)
    count_images_and_qrs("synt", DATASET_SYNT)

    # Results of the experiments ---------------------------------------------------------------------------------------
    print("Results:")
    print(f" - Number of localized qrs:\t\t\t\t\t\t{total_loc}")
    print(f" - Relative localized qrs:\t\t\t\t\t\t{total_loc / total}")

    t = df[df[DATASET] == DATASET_FLAT][IMAGE_ID].count()
    t_loc = df_loc[df_loc[DATASET] == DATASET_FLAT][IMAGE_ID].count()
    t_z = df[(df[DATASET] == DATASET_FLAT) & df[ZBAR]][IMAGE_ID].count()
    print(f" - Number of localized qrs in the flat dataset:\t{t_loc}")
    print(f" - Relative localized qrs in the flat dataset:\t{t_loc / t}")
    print(f" - Relative localized qrs in the flat dataset:\t{t_z / t}")

    t = df[df[DATASET] == DATASET_RAND][IMAGE_ID].count()
    t_loc = df_loc[df_loc[DATASET] == DATASET_RAND][IMAGE_ID].count()
    t_z = df[(df[DATASET] == DATASET_RAND) & df[ZBAR]][IMAGE_ID].count()
    print(f" - Number of localized qrs in the rand dataset:\t{t_loc}")
    print(f" - Relative localized qrs in the rand dataset:\t{t_loc / t}")
    print(f" - Relative localized qrs in the rand dataset:\t{t_z / t}")

    t = df[df[DATASET] == DATASET_SYNT][IMAGE_ID].count()
    t_loc = df_loc[df_loc[DATASET] == DATASET_SYNT][IMAGE_ID].count()
    t_z = df[(df[DATASET] == DATASET_SYNT) & df[ZBAR]][IMAGE_ID].count()
    print(f" - Number of localized qrs in the synt dataset:\t{t_loc}")
    print(f" - Relative localized qrs in the synt dataset:\t{t_loc / t}")
    print(f" - Relative localized qrs in the synt dataset:\t{t_z / t}")

    t = df[df[DATASET] == DATASET_FLAT][IMAGE_ID].count()

    def print_num_reads(name: str, num_reads: int) -> None:
        print(f" - Number of read qrs by {name}:\t\t\t\t\t{num_reads}")
        print(f" - Relative read qrs by {name}:\t\t\t\t\t{num_reads / total}")
        print(f" - Relative localized read qrs by {name}:\t\t\t{num_reads / total_loc}")

    num_reads_zbar = df[ZBAR].sum()
    print(f" - Number of read qrs by zbar:\t\t\t\t\t{num_reads_zbar}")
    print(f" - Relative read qrs by zbar:\t\t\t\t\t{num_reads_zbar / total}")
    num_reads_zbar = df_loc[ZBAR].sum()
    print(f" - Relative localized read qrs by zbar:\t\t\t{num_reads_zbar / total_loc}")
    del num_reads_zbar

    print_num_reads("AFF", df[READ_AFF].sum())
    print_num_reads("PRO", df[READ_PRO].sum())
    print_num_reads("CYL", df[READ_CYL].sum())
    print_num_reads("TPS", df[READ_TPS].sum())

    print(f" - Relative read qrs by PRO in flat:\t\t\t\t\t{df[df[DATASET] == DATASET_FLAT][READ_PRO].sum() / t}")

    def plot_bars_reads(df_base: pd.DataFrame, filename: str, zbar: bool = True) -> None:
        total_base = df_base[IMAGE_ID].count()
        values = []
        index = copy.deepcopy(corrections_labels)
        for correction in corrections:
            col_id = f"read_{correction.name.lower()}"
            values.append(df_base[df_base[col_id] == True][col_id].count() / total_base)
        if zbar:
            values.append(df_base[ZBAR].sum() / total_base)
            index += ["ZBAR"]
        dict_aux = {
            "QRs localizable by our framework": values
        }
        df_aux = pd.DataFrame(dict_aux, index=index)

        fig, ax = plt.subplots()
        df_aux.plot.bar(ax=ax)
        ax.get_legend().remove()
        ax.set_ylim(0, 1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation="horizontal")
        ax.set_yticklabels(list(map(lambda n: f"{int(n)}%", ax.get_yticks() * 100)))
        file_path = RESULTS_DIR / f"{filename}.png"
        fig.savefig(str(file_path), bbox_inches='tight', pad_inches=0)
        print(f"   -> Saved to the file {file_path}")
        plt.close(fig)

    def plot_points_reads(df_base: pd.DataFrame, filename: str, zbar: bool = True) -> None:
        total_base = df_base[IMAGE_ID].count()
        values = []
        index = copy.deepcopy(corrections_labels)
        for correction in corrections:
            col_id = f"read_{correction.name.lower()}"
            values.append(df_base[df_base[col_id] == True][col_id].count() / total_base)
        if zbar:
            values.append(df_base[ZBAR].sum() / total_base)
            index += ["ZBAR"]
        dict_aux = {
            "QRs localizable by our framework": values
        }
        df_aux = pd.DataFrame(dict_aux, index=index)

        lis = []
        points = []
        defos = [Deformation.AFFINE, Deformation.PERSPECTIVE, Deformation.CYLINDRIC, Deformation.SURFACE]
        for i, defo in enumerate(defos):
            df_defo = df_base[df_base[DEFORMATION] == defo]
            total_def = df_defo[IMAGE_ID].count()
            for j, correction in enumerate(corrections):
                col_id = f"read_{correction.name.lower()}"
                points.append([i + 1, j + 1])
                lis.append(df_defo[col_id].sum() / total_def)

        points = np.array(points)
        lis1 = np.array(lis)
        print(lis1)
        lis = np.exp(lis1 * 5 + 1) * 6
        print(lis)

        fig, ax = plt.subplots()
        # df_aux.plot.bar(ax=ax)
        ax.scatter(*points.T, s=lis, c=lis1*0.4 + 0.6, cmap='winter', alpha=0.9)
        ax.set_ylim(0, 5)
        ax.set_xlim(0, 5)
        ax.set_xticks(range(1, 5))
        ax.set_yticklabels(corrections_labels)
        ax.set_xlabel("Deformations")
        ax.set_yticks(range(1, 5))
        ax.set_xticklabels(["Affine", "Projective", "Cylindrical", "Random"])
        ax.set_ylabel("Corrections")

        for label, x, y in zip(map(lambda x: f"{x}%", lis1.round(2)*100), points[:, 0], points[:, 1]):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-4, 4),
                textcoords='offset points', ha='right', va='bottom',
                # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.9)#, alpha=0.5),
                #arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )

        # ax.get_legend().remove()
        # ax.set_ylim(0, 1)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation="horizontal")
        file_path = RESULTS_DIR / f"{filename}.png"
        fig.savefig(str(file_path), bbox_inches='tight', pad_inches=0)
        print(f"   -> Saved to the file {file_path}")
        plt.close(fig)

    def plot_rel_error_by_method(df_base: pd.DataFrame, filename: str, logy: bool = True) -> None:
        ids = [f"rel_errors_{correction.name.lower()}" for correction in corrections]
        bins = [0, 0.05, 0.1, 0.2, 0.4, 0.6, 1]
        cuts = [
            pd.cut(
                df_base[id].dropna(),
                bins=bins,
                include_lowest=True
            ).value_counts(sort=False)
            for id in ids
        ]
        df_aux = pd.concat(cuts, axis=1)
        df_aux.columns = corrections_labels
        fig, ax = plt.subplots()
        df_aux.plot.bar(ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation="horizontal")
        if logy:
            ax.set_yscale("log")

        file_path = RESULTS_DIR / f"{filename}.png"
        fig.savefig(str(file_path), bbox_inches='tight', pad_inches=0)
        print(f"   -> Saved to the file {file_path}")
        plt.close(fig)

    def plot_rel_error_by_method2(df_base: pd.DataFrame, filename: str, logy: bool = True) -> None:
        ids = [f"rel_errors_{correction.name.lower()}" for correction in corrections]
        bins = [0, 0.05, 0.1, 0.2, 0.4, 0.6, 1]
        fig, ax = plt.subplots()
        ax.hist(df_base[ids].values, bins=bins)
        if logy:
            ax.set_yscale("log")

        file_path = RESULTS_DIR / f"{filename}.png"
        fig.savefig(str(file_path), bbox_inches='tight', pad_inches=0)
        print(f"   -> Saved to the file {file_path}")
        plt.close(fig)

    def plot_rel_error_by_method3(df_base: pd.DataFrame, filename: str, logy: bool = True) -> None:
        ids = [f"rel_errors_{correction.name.lower()}" for correction in corrections]
        fig, ax = plt.subplots()
        n, b, _ = ax.hist(df_base[ids[3]].values, bins=np.linspace(0, 1, 50), histtype="step")
        if logy:
            ax.set_yscale("log")

        # fig, ax = plt.subplots()
        # ax.plot(b[:-1], n)
        # if logy:
        #     ax.set_yscale("log")

        file_path = RESULTS_DIR / f"{filename}.png"
        # fig.savefig(str(file_path), bbox_inches='tight', pad_inches=0)
        print(f"   -> Saved to the file {file_path}")
        # plt.close(fig)

    def plot_rel_error_by_method4(df_base: pd.DataFrame, filename: str, logy: bool = True) -> None:
        ids = [f"rel_errors_{correction.name.lower()}" for correction in corrections]
        num_bins = 50
        bins = np.linspace(0, 0.5, num_bins)
        cuts = [
            pd.cut(
                df_base[id].dropna(),
                bins=bins,
                include_lowest=True
            ).value_counts(sort=False)
            for id in ids
        ]
        df_aux = pd.concat(cuts, axis=1)
        df_aux.index = bins[:-1]
        df_aux.columns = corrections_labels

        ticks = np.linspace(0, num_bins, 6)
        df_aux.plot.bar(subplots=True, sharex=True, width=1, logy=True, ylim=(0.5, 1001), xticks=ticks, figsize=(7, 9), title=["", "", "", ""])
        plt.xticks(ticks, ["0%", "10%", "20%", "30%", "40%", "50%"], rotation="horizontal")
        plt.xlabel("Ratio of QR Pixels failed")

        file_path = RESULTS_DIR / f"{filename}.png"
        plt.savefig(str(file_path), bbox_inches='tight', pad_inches=0)
        print(f"   -> Saved to the file {file_path}")
        plt.close()

    print(" - Figures TODO:")
    plot_points_reads(df_loc, "points_reads")
    plot_bars_reads(df, "read_with_loc")
    plot_bars_reads(df_loc, "read_without_loc")
    plot_bars_reads(df_loc[df_loc[DEFORMATION] == Deformation.AFFINE], "defaff")
    plot_bars_reads(df_loc[df_loc[DEFORMATION] == Deformation.PERSPECTIVE], "defpro")
    plot_bars_reads(df_loc[df_loc[DEFORMATION] == Deformation.CYLINDRIC], "defcyl")
    plot_bars_reads(df_loc[df_loc[DEFORMATION] == Deformation.SURFACE], "deftps")
    print()

    print(" - Figures TODO:")
    # plot_rel_error_by_method(df, "rel_errors")
    # plot_rel_error_by_method(df[df[DEFORMATION] == Deformation.AFFINE], "rel_errors_daff")
    # plot_rel_error_by_method(df[df[DEFORMATION] == Deformation.PERSPECTIVE], "rel_errors_dpro")
    # plot_rel_error_by_method(df[df[DEFORMATION] == Deformation.CYLINDRIC], "rel_errors_dcyl")
    # plot_rel_error_by_method(df[df[DEFORMATION] == Deformation.SURFACE], "rel_errors_dtps")
    # plot_rel_error_by_method3(df, "rel_errors")
    # plot_rel_error_by_method3(df[df[DEFORMATION] == Deformation.AFFINE], "rel_errors_daff")
    # plot_rel_error_by_method3(df[df[DEFORMATION] == Deformation.PERSPECTIVE], "rel_errors_dpro")
    # plot_rel_error_by_method3(df[df[DEFORMATION] == Deformation.CYLINDRIC], "rel_errors_dcyl")
    # plot_rel_error_by_method3(df[df[DEFORMATION] == Deformation.SURFACE], "rel_errors_dtps")
    plot_rel_error_by_method4(df, "rel_errors")
    print()

    print(" - Descriptive table of the relative error:\n")
    ids = [f"rel_errors_{correction.name.lower()}" for correction in corrections]
    df_aux = pd.DataFrame(
        [df[ids].mean().values, df[ids].std().values],
        columns=corrections_labels,
        index=["Mean of the relative error", "Std. Dev. of the relative error"]
    )
    df_aux = df_aux.round(4) * 100
    print(df_aux)
    file_path = RESULTS_DIR / "rel_errors.csv"
    df_aux.to_csv(str(file_path))
    print(f"\n   -> Saved to the file {file_path}")
    del file_path
    del df_aux
    del ids

    plt.show()
