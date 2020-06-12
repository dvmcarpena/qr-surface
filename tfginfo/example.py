from pathlib import Path

import imageio
import numpy as np

from tfginfo import QRCode, Correction

image_path = Path(__file__).parent.parent / "datasets" / "colorsensing/images/IMG_20191225_202803.jpg"
image: np.ndarray = imageio.imread(image_path)

for qr in QRCode.from_image(image):
    qr.correct(method=Correction.PROJECTIVE)
    data = qr.sample().decode()

    print(data)
    qr.plot(show=True)
