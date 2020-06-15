import qrcode
from qrcode.image.pil import PilImage

# Parameters used
DATA = "https://www.color-sensing.com"
VERSION = 7
ERROR_CORRECTION = qrcode.constants.ERROR_CORRECT_H
BORDER = 0
BOX_SIZE = 1

# Creation of the QR Code
qr = qrcode.QRCode(
    version=VERSION,
    error_correction=ERROR_CORRECTION,
    box_size=BOX_SIZE,
    border=BORDER,
    image_factory=PilImage
)
qr.add_data(DATA)
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white")
img.save("qr.png")

