import numpy as np

from ..utils import rgb2binary
from .alignmentpatterns import find_alignment_patterns
from .finderpatterns import find_finder_patterns
from .models import Features


def find_all_features(image: np.ndarray, **kwargs) -> Features:
    """
    Find all features in an image, and save it in the Features object

    :param image: Image where we want to find the features
    :param kwargs: Keyword arguments for the finder functions

    :return: Structure describing all the features found
    """
    bw_image = rgb2binary(image)
    return Features(
        image=image,
        bw_image=bw_image,
        finder_patterns=find_finder_patterns(bw_image, **kwargs),
        alignment_patterns=find_alignment_patterns(bw_image, **kwargs)
    )
