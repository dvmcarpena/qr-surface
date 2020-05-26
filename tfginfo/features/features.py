from tfginfo.utils import Image, rgb2binary

from .alignmentpatterns import find_alignment_patterns
from .finderpatterns import find_finder_patterns
from .models import Features


def find_all_features(image: Image, **kwargs) -> Features:
    bw_image = rgb2binary(image)
    return Features(
        image=image,
        bw_image=bw_image,
        finder_patterns=find_finder_patterns(bw_image, **kwargs),
        alignment_patterns=find_alignment_patterns(bw_image, **kwargs)
    )
