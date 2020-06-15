import copy
from pprint import pprint
from typing import List

import pandas as pd
from matplotlib import pyplot as plt

from tfg.qrsurface import Correction, QRErrorId
from tfg.datasets import LabeledImage

LOCALIZATION = "loc"
DATASET = "dataset"
DATASET_FLAT = "colorsensing"
DATASET_RAND = "colorsensing2"
DATASET_SYNT = "synthetic_small"
ZBAR = "zbar"
READ_AFF = "read_affine"
READ_PRO = "read_projective"
READ_CYL = "read_cylindrical"
READ_TPS = "read_tps"


def read_results(target_images: List[LabeledImage], corrections: List[Correction]) -> None:
    # relative: bool = False
    # relative_percent: bool = True
    # precision: int = 4
    # localization_errors = [
    #     QRErrorId.ERROR_FEATURES,
    #     QRErrorId.NOT_ENOUGH_QRS,
    #     QRErrorId.WRONG_VERSION
    # ]
    # correction_errors = [
    #     QRErrorId.CANT_READ,
    #     QRErrorId.BAD_DATA,
    #     QRErrorId.WRONG_PIXELS
    # ]
    # localization = {
    #     err_id: 0
    #     for err_id in map(lambda e: e.name, localization_errors)
    # }
    # results = {
    #     correction: {
    #         err_id: 0
    #         for err_id in map(lambda e: e.name, correction_errors)
    #     }
    #     for correction in map(lambda e: e.name, corrections)
    # }
    #
    # for correction in map(lambda e: e.name, corrections):
    #     results[correction]["GOOD"] = 0

    # Building the DataFrame with all the data -------------------------------------------------------------------------

    d = {}
    d["image_id"] = [
        labeled_image.image_id
        for labeled_image in target_images
        for _ in range(labeled_image.num_qrs)
    ]
    d[DATASET] = [
        labeled_image.dataset
        for labeled_image in target_images
        for _ in range(labeled_image.num_qrs)
    ]
    d[LOCALIZATION] = [
        labeled_image.localization_error is None
        for labeled_image in target_images
        for _ in range(labeled_image.num_qrs)
    ]
    d["loc_error"] = [
        labeled_image.localization_error.name if labeled_image.localization_error is not None else None
        for labeled_image in target_images
        for _ in range(labeled_image.num_qrs)
    ]
    d[ZBAR] = [
        labeled_image.zbar and labeled_image.zbar_error is None
        for labeled_image in target_images
        for _ in range(labeled_image.num_qrs)
    ]
    read = {
        correction: []
        for correction in map(lambda e: e.name, corrections)
    }
    perfect = copy.deepcopy(read)
    num_errors = copy.deepcopy(read)
    rel_errors = copy.deepcopy(read)
    for labeled_image in target_images:
        if labeled_image.num_qrs == 0:
            print("ZERO", labeled_image.image_id)
        if labeled_image.localization_error is None and labeled_image.num_qrs != 0:
            assert len(labeled_image.qrs) == labeled_image.num_qrs

            for i, qr in enumerate(labeled_image.qrs):
                assert len(qr.correction_error.keys()) == 5
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

    df = pd.DataFrame(d)
    # print(df)

    # Analysing the results and producing outputs ----------------------------------------------------------------------

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     # print(df.describe())

    total = df["image_id"].count()
    total_loc = df[LOCALIZATION].sum()

    print("Parameters of the experiments:")
    print(f" - Number of images:\t\t\t\t\t\t\t{total}")

    num_images_flat = df[df[DATASET] == DATASET_FLAT][DATASET].count()
    print(f" - Number of images in the flat dataset:\t\t{num_images_flat}")

    num_images_rand = df[df[DATASET] == DATASET_RAND][DATASET].count()
    print(f" - Number of images in the random dataset:\t\t{num_images_rand}")

    num_images_synt = df[df[DATASET] == DATASET_SYNT][DATASET].count()
    print(f" - Number of images in the synthetic dataset:\t{num_images_synt}")

    print("Results:")
    print(f" - Number of localized images:\t\t\t\t\t{total_loc}")
    print(f" - Relative localized images:\t\t\t\t\t{total_loc / total}")

    num_reads_zbar = df[ZBAR].sum()
    print(f" - Number of read images by zbar:\t\t\t\t{num_reads_zbar}")
    print(f" - Relative read images by zbar:\t\t\t\t{num_reads_zbar / total}")

    num_reads_aff = df[READ_AFF].sum()
    print(f" - Number of read images by AFF:\t\t\t\t{num_reads_aff}")
    print(f" - Relative read images by AFF:\t\t\t\t\t{num_reads_aff / total}")

    num_reads_pro = df[READ_PRO].sum()
    print(f" - Number of read images by PRO:\t\t\t\t{num_reads_pro}")
    print(f" - Relative read images by PRO:\t\t\t\t\t{num_reads_pro / total}")

    num_reads_cyl = df[READ_CYL].sum()
    print(f" - Number of read images by CYL:\t\t\t\t{num_reads_cyl}")
    print(f" - Relative read images by CYL:\t\t\t\t\t{num_reads_cyl / total}")

    num_reads_tps = df[READ_TPS].sum()
    print(f" - Number of read images by TPS:\t\t\t\t{num_reads_tps}")
    print(f" - Relative read images by TPS:\t\t\t\t\t{num_reads_tps / total}")

    # h = {
    #     correction.name[:3]: []
    #     for correction in corrections
    # }
    # df_loc = df[df[LOCALIZATION]]
    # for correction in corrections:
    #     id = correction.name.lower()
    #     col_id = f"read_{id}"
    #     nread_c = df_loc[df_loc[col_id] == False][col_id].count()
    #     col_id = f"perfect_{id}"
    #     perf_c = df_loc[df_loc[col_id] == True][col_id].count()
    #     rest_c = total_loc - nread_c - perf_c
    #     h[correction.name[:3]] = [nread_c / total_loc, rest_c / total_loc, perf_c / total_loc]
    # # print(h)
    # df3 = pd.DataFrame(h, index=["Fail", "Read", "Read and without errors"])
    #
    # df_loc = df[df[LOCALIZATION]]
    # for correction in corrections:
    #     id = correction.name.lower()
    #     col_id = f"read_{id}"
    #     nread_c = df_loc[df_loc[col_id] == False][col_id].count()
    #     rest_c = total - nread_c
    #     h[correction.name[:3]] = [nread_c, rest_c]
    # h["ZBAR"] = [df["zbar"].sum(), total - df["zbar"].sum()]
    # # print(h)
    # df3 = pd.DataFrame(h, index=["Fail", "Read"])

    df_loc = df[df[LOCALIZATION]]
    h = {}
    h2 = {}
    r = []
    r2 = []
    l = []
    for correction in corrections:
        id = correction.name.lower()
        col_id = f"read_{id}"
        nread_c = df_loc[df_loc[col_id] == True][col_id].count()
        r.append(nread_c / total)
        r2.append(nread_c / total_loc)
        l.append(correction.name[:3])
    r.append(df["zbar"].sum() / total)
    r2.append(df_loc["zbar"].sum() / total_loc)
    l.append("ZBAR")
    h["All QRs"] = r
    h2["QRs localizable by our framework"] = r2
    # print(h)
    df3 = pd.DataFrame(h, index=l)
    df4 = pd.DataFrame(h2, index=l)

    # zbar_col = "zbar"
    # print(f"zbar: {df[df[zbar_col] == True][zbar_col].count()}")

    labels = []
    for correction in corrections:
        id = correction.name.lower()
        col_id = f"perfect_{id}"
        # print(f"{id}: {df[col_id].sum()}")
        labels.append(correction.name[:3])

    ids = [f"rel_errors_{correction.name.lower()}" for correction in corrections]
    bins = [0, 0.05, 0.1, 0.2, 0.4, 0.6, 1]
    cuts = [
        pd.cut(
            df[id].dropna(),
            bins=bins,
            include_lowest=True
        ).value_counts(sort=False)
        for id in ids
    ]
    df2 = pd.concat(cuts, axis=1)
    df2.columns = labels
    fig, ax = plt.subplots()
    df2.plot.bar(ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation="horizontal")
    # ax.set_yscale("log")
    # ax.hist(df[ids].values, bins=bins)
    # ax.set_yscale("log")

    # for correction in corrections:
    #     df3.plot.pie(y=correction.name[:3])
    # df3.plot.pie(y="ZBAR")
    # from matplotlib.figure import Axes
    fig, ax = plt.subplots()
    df3.plot.bar(ax=ax)
    ax.get_legend().remove()
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation="horizontal")
    fig.savefig('data/results/read_with_loc.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    fig, ax = plt.subplots()
    df4.plot.bar(ax=ax)
    ax.get_legend().remove()
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation="horizontal")
    # fig.savefig('data/results/read_without_loc.png', bbox_inches='tight', pad_inches=0)
    # fig.savefig('data/results/defaff.png', bbox_inches='tight', pad_inches=0)
    # fig.savefig('data/results/defpro.png', bbox_inches='tight', pad_inches=0)
    # fig.savefig('data/results/defcyl.png', bbox_inches='tight', pad_inches=0)
    fig.savefig('data/results/deftps.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    ids = [f"rel_errors_{correction.name.lower()}" for correction in corrections]

    df5 = pd.DataFrame(
        [df[ids].mean().values, df[ids].std().values],
        columns=[correction.name[:3] for correction in corrections],
        index=["Mean of the relative error", "Std. Dev. of the relative error"]
    )
    df5 = df5.round(4)
    filename = "data/results/rel_errors.csv"
    df5.to_csv(filename)
    print(" - Descriptive table of the relative error by method:\n")
    print(df5)
    print(f"\n   -> Saved to the file {filename}")

    plt.show()


