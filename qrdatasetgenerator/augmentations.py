import json
from pathlib import Path

import imageio
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
from skimage import transform, img_as_ubyte
from tqdm import tqdm


def img_aug(image: np.array, targets: np.array) -> np.array:
    """
    Performs random augmentations on the image using imgaug's augmentations.

    :param image: Image to be augmented
    :param targets: Bounding boxes. Format of the targets: [[x, y, width, height]]

    :return: A new augmented and the bounding box containing th augmented QR Code
    """
    img_x, img_y = image.shape[1], image.shape[2]
    bbs = BoundingBoxesOnImage([
        BoundingBox(
            x1=(target[0] - target[2]/2)*img_x,  # (x - w/2) * size_x
            x2=(target[0] + target[2]/2)*img_x,  # (x + w/2) * size_x
            y1=(target[1] - target[3]/2)*img_y,  # (y - h/2) * size_y
            y2=(target[1] + target[3]/2)*img_y   # (y + h/2) * size_y
        )
        for target in targets
    ], shape=(img_x, img_y))

    seq = iaa.Sequential([
        # iaa.Invert(0.2, per_channel=True),
        iaa.GammaContrast(gamma=(0.8, 1.2)),
        # iaa.Fliplr(0.2),  # flip horizontally 20% of the images
        iaa.Rotate((-20, 20), mode='edge'),
        iaa.PerspectiveTransform(scale=(0.01, 0.10), mode="replicate", fit_output=True),
        iaa.Affine(translate_percent={"x": (-0.10, 0.10), "y": (-0.10, 0.10)},
                   scale=(0.80, 1.2), rotate=(0, 20), mode='edge', fit_output=True)
    ])

    aug_img, aug_bbs = seq(image=image, bounding_boxes=bbs)
    aug_img = img_as_ubyte(transform.resize(aug_img, image.shape, order=0))

    for id, target in enumerate(targets):
        max_x, min_x = max(aug_bbs[id].x1, aug_bbs[id].x2, 0), min(aug_bbs[id].x1, aug_bbs[id].x2, img_x)
        max_y, min_y = max(aug_bbs[id].y1, aug_bbs[id].y2, 0), min(aug_bbs[id].y1, aug_bbs[id].y2, img_y)
        x, y, w, h = (max_x + min_x)/2, (max_y + min_y)/2, max_x - min_x, max_y - min_y
        target[:] = x/img_x, y/img_y, w/img_x, h/img_y

    return aug_img, targets


# Parameters of the augmentation
NUM_AUGS = 20

# Creation of target folders
images_folder = Path("dataset/images")
qr_labels = Path("dataset/annotations")
yolo_labels = Path("dataset/yolo")
base_images = list(images_folder.iterdir())
aug_images = Path("dataset/images")
aug_images.mkdir(exist_ok=True, parents=True)

for p in tqdm(base_images):
    # Read the template images
    image = np.array(imageio.imread(str(p)))
    label = (yolo_labels / f"{p.stem}.txt").read_text()
    bbox = list(map(float, label.split(" ")[1:]))

    i = 0
    while i < NUM_AUGS:
        try:
            # For each template image do aygmentations and save them with metadata
            new_image, result_bbox = img_aug(image, [bbox])
            augname = f"{p.stem}_{i}"
            imageio.imwrite(str(aug_images / f"{augname}.jpg"), new_image, format="JPEG-PIL", quality=95)

            orig_dict = json.loads((qr_labels / f"{p.stem}.json").read_text())
            image_dict = {
                "id": augname,
                "id_original": orig_dict["id_original"],
                "deformation": "perspective",
                "num_qrs": orig_dict["num_qrs"]
            }
            (qr_labels / f"{augname}.json").write_text(json.dumps(image_dict))
            i += 1
        except ValueError as e:
            pass
