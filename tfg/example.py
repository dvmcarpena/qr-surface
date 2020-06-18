from pathlib import Path

import imageio
import numpy as np

from tfg import QRCode, Correction

image_path = Path(__file__).parent.parent / "data/datasets/flat/images/IMG_20191225_202803.jpg"
image: np.ndarray = imageio.imread(image_path)

for qr in QRCode.from_image(image):
    qr.correct(method=Correction.PROJECTIVE)
    data = qr.sample().decode()

    print(data)
    qr.plot(show=True)
