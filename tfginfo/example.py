from pathlib import Path

import imageio
import numpy as np

from tfginfo import QRCode, Correction

images_folder = Path(__file__).parent.parent / "images"
image_path = images_folder / "v7_colorsensing/projective/1/good/IMG_20191225_202803.jpg"



image: np.ndarray = imageio.imread(image_path)

for qr in QRCode.from_image(image):
    qr.correct(method=Correction.PROJECTIVE)
    data = qr.decode()

    print(data)
    qr.plot(show=True)
