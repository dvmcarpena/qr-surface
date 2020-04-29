from tfginfo.utils import Image

from .alignmentpatterns import find_alignment_patterns
from .finderpatterns import find_finder_patterns
from .models import Features


def find_all_features(image: Image, **kwargs) -> Features:
    return Features(
        image=image,
        finder_patterns=find_finder_patterns(image, **kwargs),
        alignment_patterns=find_alignment_patterns(image, **kwargs)
    )
