from pathlib import Path
import traceback

import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

from tfginfo.utils import Image
from tfginfo.features import Features
from tfginfo.decode import decode
from tfginfo.qr import QRCode, Correction
from tfginfo.error import QRErrorId, QRException
from tfginfo.images import LabeledImage, parse_labeled_images, parse_original_qrs, Deformation

import warnings
warnings.filterwarnings('error')


if __name__ == "__main__":
    images_dir = (Path(__file__).parent.parent / "images" / "labeled").resolve()

    def filter_func(labeled_image: LabeledImage) -> bool:
        return (
            # labeled_image.version == 2
            labeled_image.version == 7
            and labeled_image.method == Deformation.PROJECTIVE
            and labeled_image.num_qrs == 1
            # and labeled_image.has_error()
            and not labeled_image.has_error()
        )

    original_qrs = parse_original_qrs(images_dir)
    target_images = parse_labeled_images(
        images_dir,
        filter_func=filter_func
    )

    with tqdm(total=len(target_images), ncols=150) as progress_bar:
        for labeled_image in target_images:
            short_path = labeled_image.path.relative_to(images_dir)
            progress_bar.set_description(f"{short_path}\n", refresh=False)

            image: Image = imageio.imread(str(labeled_image.path))

            try:
                if labeled_image.has_data:
                    try:
                        results = decode(image)
                        # print(f"ZBar read {len(results)} messages:")
                        for message in results:
                            # print(f"\t{message}")
                            pass
                    except ValueError:
                        # print("No QR read with ZBar")
                        pass

                features = Features.from_image(image)

                try:
                    qrs = list(QRCode.from_features(image, features))
                except Exception:
                    features.plot()
                    raise QRErrorId.ERROR_FEATURES.exception()

                if len(qrs) < labeled_image.num_qrs:
                    features.plot()
                    raise QRErrorId.NOT_ENOUGH_QRS.exception()
                elif len(qrs) == 0:
                    features.plot()

                for qr in qrs:
                    if labeled_image.version != qr.version:
                        features.plot()
                        raise QRErrorId.WRONG_VERSION.exception(
                            estimated=qr.version,
                            expected=labeled_image.version
                        )

                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                    fig.suptitle(labeled_image.path.name)

                    features.plot(axes=ax1)

                    # import numpy as np
                    # qr.fourth_corner = np.array([1009, 620])

                    # from skimage import img_as_ubyte, color, filters, transform
                    # from tfginfo.features.corner import corner_scan
                    # gray_image: np.ndarray = color.rgb2gray(qr.image)
                    # threshold: np.ndarray = filters.threshold_sauvola(gray_image, 151)
                    # bw_image: np.ndarray = gray_image > threshold
                    # qr.fourth_corner = corner_scan(
                    #     bw_image,
                    #     2,
                    #     corner_radius=0,
                    #     fuzzy_radius=1,
                    #     blank_radius=1
                    # )[::-1]

                    qr.plot(axes=ax2)

                    qr.correct(method=Correction.PROJECTIVE)#, border=5)
                    qr.plot(axes=ax3)

                    # from tfginfo.transformations.ideal import IdealQRCode
                    # from tfginfo.matching import MatchingFeatures
                    #
                    # iqr = IdealQRCode(qr.version, 10, 20)
                    # f = [
                    #     MatchingFeatures.FINDER_CENTERS,
                    #     MatchingFeatures.FINDER_CORNERS,
                    #     MatchingFeatures.ALIGNMENTS_CENTERS,
                    #     MatchingFeatures.FOURTH_CORNER
                    # ]
                    # references = qr.create_references(f)
                    # ax3.scatter(*iqr.get_references(references).T)

                    if labeled_image.has_data:
                        try:
                            data = qr.decode(bounding_box=False)
                        except ValueError:
                            raise QRErrorId.CANT_READ.exception()

                        # print(data)

                        original_qr = original_qrs[labeled_image.original_id]
                        sampled_qr = qr.sample()

                        # from tfginfo.qr import SampledQRCode
                        # import copy
                        # qr_sampled = copy.deepcopy(qr)
                        # qr_sampled.binarize()
                        # qr_sampled.correct(method=Correction.PROJECTIVE, bitpixel=1, border=0)
                        #
                        # sampled_qr = SampledQRCode(
                        #     image=qr_sampled.image,
                        #     version=qr_sampled.version
                        # )
                        # sampled_qr.plot(axes=ax4)
                        # from skimage import transform, img_as_ubyte
                        # ax4.imshow(transform.rescale(original_qr, 3, multichannel=True, order=0,
                        #                              anti_aliasing=False), interpolation="none")
                        # # ax4.imshow(original_qr, interpolation="none")
                        #
                        #
                        # from tfginfo.transformations.ideal import IdealQRCode
                        # from tfginfo.matching import MatchingFeatures
                        # import numpy as np
                        # iqr = IdealQRCode(qr.version, 3, 5)
                        # f = [
                        #     MatchingFeatures.FINDER_CENTERS,
                        #     MatchingFeatures.FINDER_CORNERS,
                        #     MatchingFeatures.ALIGNMENTS_CENTERS,
                        #     MatchingFeatures.FOURTH_CORNER
                        # ]
                        # references = qr.create_references(f)
                        # points = iqr.get_references(references)
                        # points -= np.array([5, 5]) * 3
                        # ax4.scatter(*points.T)
                        sampled_qr.plot_differences(original_qr, axes=ax4)

                        errors = sampled_qr.count_errors(original_qr)
                        if errors > 0:
                            raise QRErrorId.WRONG_PIXELS.exception(num_pixels=errors)

                    plt.close()
                    # plt.show()

                if labeled_image.error_id is None:
                    destination = labeled_image.path.parent
                else:
                    destination = labeled_image.path.parent.parent.parent / "good"
            except QRException as e:
                # print()
                # print(e)
                # print(labeled_image.path.relative_to(labeled_images_dir))
                # print(labeled_image.path.name, e.id)

                # plt.close()
                plt.show()

                if labeled_image.error_id is None:
                    destination = labeled_image.path.parent.parent / "bad" / e.id
                else:
                    destination = labeled_image.path.parent.parent / e.id
            except Exception as e:
                # print(labeled_image.path.relative_to(images_dir))
                # print("Unexpected exception")
                # traceback.print_exc()
                raise e

            dest_dir = destination.resolve()
            destination = dest_dir / labeled_image.path.name
            if labeled_image.path != destination:
                print()
                print(f"{labeled_image.path.relative_to(images_dir)} -> {destination.relative_to(images_dir)}")

                # dest_dir.mkdir(parents=True, exist_ok=True)
                # if not destination.exists():
                #     labeled_image.path.replace(destination)
                # else:
                #     raise ValueError("owerwriting stoped")

            progress_bar.update()

    # plt.show()
