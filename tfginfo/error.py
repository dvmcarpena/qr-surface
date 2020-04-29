from enum import Enum, unique


@unique
class QRErrorId(Enum):
    NOT_ENOUGH_QRS = "Not enough qrs detected"
    ERROR_FEATURES = "Error at from_features"
    WRONG_VERSION = "Estimated version {estimated}, real one {expected}"
    CANT_READ = "Cant read the QR Code"
    WRONG_PIXELS = "Found {num_pixels} pixels with errors"

    def exception(self, **kwargs) -> 'QRException':
        raise QRException(self, **kwargs)


class QRException(Exception):

    def __init__(self, error_id: QRErrorId, **kwargs):
        super(QRException, self).__init__()
        self.id = error_id.name
        self.description = error_id.value.format(**kwargs)

    def __str__(self) -> str:
        return f"{self.id}: {self.description}"
