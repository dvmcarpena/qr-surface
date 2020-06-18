from typing import Dict

import numpy as np

from ..matching import MatchingFeatures, References
from ..utils import get_alignments_centers, get_size_from_version


class IdealQRCode:
    """
    A ideal QR Code collection of patterns, with the points that correspond to all the features that we could find
    in the localization. Used to create the destiny references
    """

    def __init__(self, version: int, bitpixel: int, border: int):
        assert isinstance(version, int)
        assert isinstance(bitpixel, int)
        assert isinstance(border, int)

        self.version = version
        self.bitpixel = bitpixel
        self.border = border

        size = get_size_from_version(self.version)
        self.size = (size + 2 * self.border) * self.bitpixel
        self.finders_centers = self._apply_dimensions(np.array([
            [3, 3],
            [size - 1 - 3, 3],
            [3, size - 1 - 3]
        ])) + np.array([bitpixel / 2]).astype(int)
        self.finders_corners = np.array([
            [0, 0],
            [7 * self.bitpixel - 1, 0],
            [7 * self.bitpixel - 1, 7 * self.bitpixel - 1],
            [0, 7 * self.bitpixel - 1],
            [(size - 7) * self.bitpixel, 0],
            [size * self.bitpixel - 1, 0],
            [size * self.bitpixel - 1, 7 * self.bitpixel - 1],
            [(size - 7) * self.bitpixel, 7 * self.bitpixel - 1],
            [0, (size - 7) * self.bitpixel],
            [7 * self.bitpixel - 1, (size - 7) * self.bitpixel],
            [7 * self.bitpixel - 1, size * self.bitpixel - 1],
            [0, size * self.bitpixel - 1]
        ]) + np.array([self.border, self.border]) * self.bitpixel
        self.finders_corners_by_finder = [
            self.finders_corners[0:4],
            self.finders_corners[4:8],
            self.finders_corners[8:]
        ]
        alignments_centers = np.array(get_alignments_centers(self.version))
        if len(alignments_centers) > 0:
            alignments_centers = alignments_centers[np.lexsort((alignments_centers[:, 0],
                                                                alignments_centers[:, 1]))]
            self.alignments_centers = self._apply_dimensions(alignments_centers) + np.array([bitpixel / 2]).astype(int)
        else:
            self.alignments_centers = np.array(alignments_centers)
        self.fourth_corner = (
            (np.array([size, size]) + np.array([self.border, self.border]))
            * self.bitpixel
            - np.array([1, 1])
        )

    def get_references(self, references: References) -> np.ndarray:
        """
        Given a object references that chooses which type of references have been selected, returns the array of
        reference points

        :param references: A selector of the type of target references

        :return: Array of reference points
        """
        match_features = self._feature_to_points(references)

        return np.concatenate([
            match_features[feature][1]
            for feature in references.features
            if match_features[feature][0]
        ], axis=0)

    def _apply_dimensions(self, points: np.ndarray) -> np.ndarray:
        return (points + np.array([self.border, self.border])) * self.bitpixel

    def _feature_to_points(self, references: References) -> Dict:
        return {
            MatchingFeatures.FINDER_CENTERS: (True, self.finders_centers),
            MatchingFeatures.FINDER_CORNERS: (True, self.finders_corners),
            MatchingFeatures.ALIGNMENTS_CENTERS: (
                any(references.alignments_found),
                self.alignments_centers[references.alignments_found]
            ),
            MatchingFeatures.FOURTH_CORNER: (references.fourth_corner_found, np.array([self.fourth_corner]))
        }
