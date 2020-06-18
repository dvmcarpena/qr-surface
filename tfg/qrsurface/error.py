from dataclasses import dataclass, field
from enum import Enum, unique


@dataclass
class BadModules:
    """
    A structure with the state obtained when a execution of decoding finishes and a result with the number of modules
    fail is generated
    """
    count: int
    relative: float
    success_rate: float = field(init=False)

    def __post_init__(self):
        self.success_rate = 1 - self.relative


@unique
class QRErrorId(Enum):
    """
    An enumeration of the possible identifier of errors in the decoding pipeline with their respective error
    descriptions
    """
    NOT_ENOUGH_QRS = "Not enough qrs detected"
    ERROR_FEATURES = "Error at from_features"
    WRONG_VERSION = "Estimated version {estimated}, real one {expected}"
    CANT_READ = "Cant read the QR Code"
    BAD_DATA = "The read data is different from the expected"
    WRONG_PIXELS = "Found {num_pixels} pixels with errors"

    def exception(self, **kwargs) -> 'QRException':
        """
        Generates an QRException for the given type of error

        :param kwargs: Keyword arguments to the QRException

        :return: A QRException that has error_id as the current object
        """
        raise QRException(self, **kwargs)

    def correction_exception(self, bad_modules: BadModules, **kwargs):
        """
        Generates an CorrectionExeception for the given type of error

        :param bad_modules: The BadModules dataclass given in the current error
        :param kwargs: Keyword arguments to the QRException

        :return: A CorrectionExeception that has error_id as the current object
        """
        raise CorrectionException(bad_modules, self, **kwargs)


class QRException(Exception):
    """
    An exception representing any error thrown by qrsurface
    """

    def __init__(self, error_id: QRErrorId, **kwargs):
        super(QRException, self).__init__()
        self.id = error_id.name
        self.error = error_id
        self.description = error_id.value.format(**kwargs)

    def __str__(self) -> str:
        return f"{self.id}: {self.description}"


class CorrectionException(QRException):
    """
    An exception that ocurred after a correction, which has a related number of bad modules associated
    """

    def __init__(self, bad_modules: BadModules, error_id: QRErrorId, **kwargs):
        super(CorrectionException, self).__init__(error_id, **kwargs)
        self.bad_modules = bad_modules
