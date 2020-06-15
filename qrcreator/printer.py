import itertools
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageCms


def draw_qr_grid(background: Image.Image,
                 qr: Image.Image,
                 x_len,
                 y_len,
                 x_offset=0,
                 y_offset=0,
                 margin=200):
    xs, ys = background.size
    qr_size = int((xs - (2 * x_offset + margin * 2 * x_len)) / x_len)
    
    xx = x_offset + margin + np.arange(0, x_len) * (margin * 2 + qr_size)
    yy = y_offset + margin + np.arange(0, y_len) * (margin * 2 + qr_size)
    positions = itertools.product(xx, yy)
    
    new_size = (qr_size, qr_size)
    qr = qr.resize(new_size)
    
    for x, y in positions:
        background.paste(qr, (x, y))


a4_size_inches = (8.3, 11.7)
dpi = 600

a4_size_pixels = tuple(map(lambda x: int(x * dpi), a4_size_inches))
qr_img = Image.open("../data/print/qr.png")

background = Image.new("RGB", a4_size_pixels, (255, 255, 255))
draw_qr_grid(background, qr_img, 3, 4)
background.save("print_big.png", dpi=(dpi, dpi))

background = Image.new("RGB", a4_size_pixels, (255, 255, 255))
draw_qr_grid(background, qr_img, 4, 5, x_offset=50, y_offset=50, margin=150)
background.save("print_small.png", dpi=(dpi, dpi))

