from itertools import product
import json
from pathlib import Path
import random
import string

import numpy as np
from PIL import Image
import qrcode
from qrcode.image.pil import PilImage
from tqdm import tqdm

images_folder = Path("dataset/images")
images_folder.mkdir(exist_ok=True, parents=True)
yolo_labels = Path("dataset/yolo")
yolo_labels.mkdir(exist_ok=True, parents=True)
qr_labels = Path("dataset/annotations")
qr_labels.mkdir(exist_ok=True, parents=True)
bitmaps = Path("dataset/bitmaps")
bitmaps.mkdir(exist_ok=True, parents=True)

QR_CAPACITY_PATH = str(Path(__file__).parent / "qr_capacity.npy")
QR_CAPACITY: np.ndarray = np.load(QR_CAPACITY_PATH)


def get_qr_capacity(qr_version: int, error_correction: str) -> int:
    if error_correction == qrcode.constants.ERROR_CORRECT_L:
        ec = 0
    elif error_correction == qrcode.constants.ERROR_CORRECT_M:
        ec = 1
    elif error_correction == qrcode.constants.ERROR_CORRECT_Q:
        ec = 2
    elif error_correction == qrcode.constants.ERROR_CORRECT_H:
        ec = 3
    else:
        raise ValueError("Invalid error correction")

    capacity = int(QR_CAPACITY[np.logical_and(QR_CAPACITY[:, 0] == qr_version,
                                              QR_CAPACITY[:, 1] == ec)][0, 4])
    return capacity


def randomword(length):
   letters = string.ascii_letters + string.digits + string.punctuation
   return ''.join(random.choice(letters) for _ in range(length))


def create_background_img(side_size: int):
    img = np.zeros([side_size, side_size, 3], dtype=np.uint8)
    img.fill(255)  # or img[:] = 255
    return Image.fromarray(img)


def insert_QR_into_background(QR_img: Image.Image, background_size: int = 416):
    """Inserts a QR code image into a background image. It is assumed that both images are squared.

    Args:
        QR_img: The image containing the QR code.

    Returns:

    """
    background_img = create_background_img(background_size)
    QR_side_size = QR_img.size[0]
    x, y = background_size // 2 - QR_side_size // 2, background_size // 2 - QR_side_size // 2
    background_img.paste(QR_img, (x, y))
    return background_img


NUM_VERSIONS = 13
NUM_CODES = 3
# DATA = "https://www.color-sensing.com"
DATA = None
# ERROR_CORRECTION = qrcode.constants.ERROR_CORRECT_H
ERROR_CORRECTION = None
BORDER = 4
BOX_SIZE = 8
FULL_CAPACITY = False
SEED = 1
# IMAGE_SIZE = 416
IMAGE_SIZE = 800

random.seed(SEED)
error_correction_levels = [
    qrcode.constants.ERROR_CORRECT_L,
    qrcode.constants.ERROR_CORRECT_M,
    qrcode.constants.ERROR_CORRECT_Q,
    qrcode.constants.ERROR_CORRECT_H
]

for index, (version, i) in tqdm(list(enumerate(product(range(1, NUM_VERSIONS + 1), range(NUM_CODES)), start=1))):
    random_ec = random.choice(error_correction_levels)
    error_correction = random_ec if ERROR_CORRECTION is None else ERROR_CORRECTION

    capacity = get_qr_capacity(version, error_correction)
    data_length = capacity - random.randint(0, capacity // 2) if not FULL_CAPACITY else capacity
    data = randomword(data_length) if DATA is None else DATA

    # Creation of the QR Code
    qr = qrcode.QRCode(
        version=version,
        error_correction=error_correction,
        box_size=1,
        border=0,
        image_factory=PilImage
    )
    qr.add_data(data)
    qr.make(fit=False)
    qr.modules[-1][-1] = True
    qr.make(fit=False)
    qr.modules[-1][-1] = True
    bitmap = qr.make_image(fill_color="black", back_color="white")
    bitmap.save(f"dataset/bitmaps/{index}.png")

    qr.border = BORDER
    qr.box_size = BOX_SIZE
    qr_image = qr.make_image(fill_color="black", back_color="white").get_image()
    image = insert_QR_into_background(qr_image, background_size=IMAGE_SIZE)

    label = f"{version-1} " \
            f"{IMAGE_SIZE // 2 / IMAGE_SIZE} " \
            f"{IMAGE_SIZE // 2 / IMAGE_SIZE} " \
            f"{(qr_image.width - 2*qr.border*qr.box_size) / IMAGE_SIZE} " \
            f"{(qr_image.height - 2*qr.border*qr.box_size) / IMAGE_SIZE}"

    label_file = yolo_labels / f"{index}.txt"
    with open(label_file, "w") as f:
        f.write(label)

    image.save(images_folder / f"{index}.jpg", "jpeg", quality=95)

    bitmap_dict = {
        "id": str(index),
        "version": version,
        "error_correction": error_correction,
        "data": data
    }
    (bitmaps / f"{index}.json").write_text(json.dumps(bitmap_dict))

    image_dict = {
        "id": str(index),
        "id_original": str(index),
        "deformation": "affine",
        "num_qrs": 1
    }
    (qr_labels / f"{index}.json").write_text(json.dumps(image_dict))
