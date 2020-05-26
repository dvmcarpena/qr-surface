from dataclasses import dataclass, field
from enum import auto, Enum, unique
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import imageio
import numpy as np
from skimage import img_as_ubyte, color, filters

from tfginfo.error import QRErrorId
from tfginfo.qr import Correction


@unique
class ErrorCorrection(Enum):
    L = 1
    M = 0
    Q = 3
    H = 2


@unique
class Deformation(Enum):
    AFFINE = auto()
    PERSPECTIVE = auto()
    CYLINDRIC = auto()
    SURFACE = auto()


@dataclass
class BadModules:
    count: int
    relative: float
    success_rate: float = field(init=False)

    def __post_init__(self):
        self.success_rate = 1 - self.relative


@dataclass
class LabeledQRCode:
    deformation: str
    correction_error: Optional[Dict[Correction, Optional[QRErrorId]]]
    bad_modules: Optional[Dict[Correction, BadModules]]


@dataclass
class LabeledImage:
    path: Path
    data_path: Path
    dataset: str
    bitmap_id: str
    version: int
    has_data: bool
    error_correction: Optional[str]
    data: Optional[str]
    image_id: str
    num_qrs: int
    localization_error: Optional[QRErrorId]
    qrs: List[LabeledQRCode]
    # deformation: str
    # correction_error: Optional[Dict[Correction, Optional[QRErrorId]]]
    # bad_modules: Optional[Dict[Correction, BadModules]]

    @classmethod
    def from_path(cls, path: Path) -> 'LabeledImage':
        dataset_dir = path.parent.parent
        dataset = dataset_dir.name

        image_id = path.stem
        data_path = dataset_dir / "annotations" / f"{image_id}.json"
        image_data = json.loads(data_path.read_text())
        assert image_data["id"] == image_id

        bitmap_id = image_data["id_original"]
        bitmap_data = json.loads((dataset_dir / "bitmaps" / f"{bitmap_id}.json").read_text())
        assert bitmap_data["id"] == bitmap_id

        num_qrs = int(image_data["num_qrs"])
        if "deformation" in image_data.keys():
            qrs_datas = image_data.get("qrs", [{"deformation": image_data["deformation"]}] * num_qrs)
        else:
            qrs_datas = image_data["qrs"]
        assert len(qrs_datas) == num_qrs

        localization_error_input: Optional[str] = image_data.get("localization_error", None)
        localization_error: Optional[QRErrorId]
        if localization_error_input is not None:
            localization_error = QRErrorId.__members__[localization_error_input]
        else:
            localization_error = None

        qrs = []
        for qr_data in qrs_datas:
            if localization_error is not None:
                correction_error = None
                bad_modules = None
            else:
                correction_error_dict: Dict[str, Optional[str]] = qr_data.get("correction_error", dict())
                correction_error = {
                    Correction.__members__[key]: QRErrorId.__members__[item] if item is not None else item
                    for key, item in correction_error_dict.items()
                }

                bad_modules_dict: Dict[str, Dict[str, Union[int, float]]] = qr_data.get("bad_modules", dict())
                bad_modules = {
                    Correction.__members__[key]: BadModules(count=item["count"], relative=item["relative"])
                    for key, item in bad_modules_dict.items()
                }
            qrs.append(LabeledQRCode(
                deformation=Deformation.__members__[qr_data["deformation"].upper()],
                correction_error=correction_error,
                bad_modules=bad_modules
            ))
        assert len(qrs) == num_qrs

        # localization_error_input: Optional[str] = image_data.get("localization_error", None)
        # localization_error: Optional[QRErrorId]
        # if localization_error_input is not None:
        #     localization_error = QRErrorId.__members__[localization_error_input]
        #     correction_error = None
        #     bad_modules = None
        # else:
        #     localization_error = localization_error_input
        #     correction_error_dict: Dict[str, Optional[str]] = image_data.get("correction_error", dict())
        #     correction_error = {
        #         Correction.__members__[key]: QRErrorId.__members__[item] if item is not None else item
        #         for key, item in correction_error_dict.items()
        #     }
        #
        #     bad_modules_dict: Dict[str, Dict[str, Union[int, float]]] = image_data.get("bad_modules", dict())
        #     bad_modules = {
        #         Correction.__members__[key]: BadModules(count=item["count"], relative=item["relative"])
        #         for key, item in bad_modules_dict.items()
        #     }

        return cls(
            path=path,
            data_path=data_path,
            dataset=dataset,
            bitmap_id=bitmap_id,
            image_id=image_id,
            # version=int(bitmap_data["version"]) if bitmap_data["version"] is not None else None,
            version=int(bitmap_data["version"]),
            has_data=(dataset_dir / "bitmaps" / f"{bitmap_id}.png").exists(),
            error_correction=bitmap_data.get("error_correction", None),
            data=bitmap_data.get("data", None),
            num_qrs=num_qrs,
            localization_error=localization_error,
            qrs=qrs,
            # deformation=Deformation.__members__[image_data["deformation"].upper()],
            # localization_error=localization_error,
            # correction_error=correction_error,
            # bad_modules=bad_modules
        )

    def has_error(self) -> bool:
        return (
            self.localization_error is not None
            or any(
                (all(items is not None for items in qr.correction_error.values())
                if qr.correction_error is not None else False)
                for qr in self.qrs
            )
        )

    def load_raw_data(self) -> Dict:
        return json.loads(self.data_path.read_text())

    def save_raw_data(self, data: Dict) -> None:
        self.data_path.write_text(json.dumps(data))

    def update_localization_error(self, localization_error: QRErrorId, update: bool = False):
        source = None
        if self.localization_error is None:
            # if self.qrs is not None:
            #     source = self.qrs
            changes = True
        elif self.localization_error != localization_error:
            changes = True
        else:
            changes = False

        if changes:
            print()
            print(f"Localization error: {source} -> {localization_error}")

            if update:
                data = self.load_raw_data()
                data["localization_error"] = localization_error.name
                for qr in data["qrs"]:
                    if "correction_error" in qr.keys():
                        qr.pop("correction_error")
                    if "bad_modules" in qr.keys():
                        qr.pop("bad_modules")
                self.save_raw_data(data)

        return changes

    def update_legacy(self):
        data = self.load_raw_data()
        if "qrs" not in data.keys():
            print("Move deformation to new qrs field")
            data["qrs"] = [{"deformation": data["deformation"]}] * self.num_qrs
        if "deformation" in data.keys():
            print("Remove old deformation")
            data.pop("deformation")
        if "error_id" in data.keys():
            print("Remove old error_id")
            data.pop("error_id")
        if "corrections" in data.keys():
            print("Remove old corrections")
            data.pop("corrections")
        if "correction_error" in data.keys():
            print("Remove old correction_error")
            data.pop("correction_error")
        if "bad_modules" in data.keys():
            print("Remove old bad_modules")
            data.pop("bad_modules")
        self.save_raw_data(data)

    def update_successfull_correction(self, correction: Correction, qr_index: int, bad_modules: BadModules, update: bool = False) -> bool:
        return self._update_correction(correction, qr_index, bad_modules, None, update=update)

    # def update_successfull_correction(self, correction: Correction, bad_modules: BadModules, update: bool = False) -> bool:
    #     return self._update_correction(correction, bad_modules, None, update=update)

    def update_correction_error(self, correction: Correction, qr_index: int, bad_modules: BadModules,
                                correction_error: Optional[QRErrorId], update: bool = False) -> bool:
        return self._update_correction(correction, qr_index, bad_modules, correction_error, update=update)

    # def update_correction_error(self, correction: Correction, bad_modules: BadModules,
    #                             correction_error: Optional[QRErrorId], update: bool = False) -> bool:
    #     return self._update_correction(correction, bad_modules, correction_error, update=update)

    def _update_correction(self, correction: Correction, qr_index: int, bad_modules: BadModules,
                           correction_error: Optional[QRErrorId], update: bool) -> bool:
        source_correction = None
        if self.localization_error is not None:
            source_correction = self.localization_error.name
        if correction_error is None:
            dest_correction = "GOOD"
        else:
            dest_correction = correction_error.name
        if self.qrs[qr_index].correction_error is None:
            changes_correction = True
        elif correction not in self.qrs[qr_index].correction_error.keys():
            changes_correction = True
        elif self.qrs[qr_index].correction_error[correction] != correction_error:
            if self.qrs[qr_index].correction_error[correction] is None:
                source_correction = "GOOD"
            else:
                source_correction = self.qrs[qr_index].correction_error[correction].name

            changes_correction = True
        else:
            changes_correction = False

        source_bad_mod = None
        if self.qrs[qr_index].bad_modules is None:
            if bad_modules is None:
                changes_bad_mod = False
            else:
                changes_bad_mod = True
        elif correction not in self.qrs[qr_index].bad_modules.keys():
            changes_bad_mod = True
        elif (self.qrs[qr_index].bad_modules[correction].count != bad_modules.count
              or self.qrs[qr_index].bad_modules[correction].relative != bad_modules.relative):
            source_bad_mod = self.qrs[qr_index].bad_modules[correction]
            changes_bad_mod = True
        else:
            changes_bad_mod = False

        if changes_correction or changes_bad_mod:
            print()

        if changes_correction:
            print(f"Correction error from {correction.name} with QR number {qr_index}: {source_correction} -> {dest_correction}")
        if changes_bad_mod:
            print(f"Bad modules from {correction.name} with QR number {qr_index}: {source_bad_mod} -> {bad_modules}")

        if (changes_correction or changes_bad_mod) and update:
            data = self.load_raw_data()
            if changes_correction:
                print("localization_error to None")
                data["localization_error"] = None
                qr = data["qrs"][qr_index]
                if "correction_error" not in qr.keys():
                    qr["correction_error"] = {}
                qr["correction_error"][correction.name] = None if correction_error is None else correction_error.name
            if changes_bad_mod:
                qr = data["qrs"][qr_index]
                if bad_modules is None:
                    qr.pop("bad_modules")
                else:
                    if "bad_modules" not in qr.keys():
                        qr["bad_modules"] = {}
                    qr["bad_modules"][correction.name] = {
                        "count": bad_modules.count,
                        "relative": bad_modules.relative
                    }
            self.save_raw_data(data)

        return changes_correction or changes_bad_mod

    # def _update_correction(self, correction: Correction, bad_modules: BadModules, correction_error: Optional[QRErrorId],
    #                        update: bool) -> bool:
    #     source_correction = None
    #     if self.localization_error is not None:
    #         source_correction = self.localization_error.name
    #     if correction_error is None:
    #         dest_correction = "GOOD"
    #     else:
    #         dest_correction = correction_error.name
    #     if self.correction_error is None:
    #         changes_correction = True
    #     elif correction not in self.correction_error.keys():
    #         changes_correction = True
    #     elif self.correction_error[correction] != correction_error:
    #         if self.correction_error[correction] is None:
    #             source_correction = "GOOD"
    #         else:
    #             source_correction = self.correction_error[correction].name
    #
    #         changes_correction = True
    #     else:
    #         changes_correction = False
    #
    #     source_bad_mod = None
    #     if self.bad_modules is None:
    #         if bad_modules is None:
    #             changes_bad_mod = False
    #         else:
    #             changes_bad_mod = True
    #     elif correction not in self.bad_modules.keys():
    #         changes_bad_mod = True
    #     elif (self.bad_modules[correction].count != bad_modules.count
    #           or self.bad_modules[correction].relative != bad_modules.relative):
    #         source_bad_mod = self.bad_modules[correction]
    #         changes_bad_mod = True
    #     else:
    #         changes_bad_mod = False
    #
    #     if changes_correction or changes_bad_mod:
    #         print()
    #
    #     if changes_correction:
    #         print(f"Correction error from {correction.name}: {source_correction} -> {dest_correction}")
    #     if changes_bad_mod:
    #         print(f"Bad modules from {correction.name}: {source_bad_mod} -> {bad_modules}")
    #
    #     if (changes_correction or changes_bad_mod) and update:
    #         data = self.load_raw_data()
    #         if changes_correction:
    #             data["localization_error"] = None
    #             if "correction_error" not in data.keys():
    #                 data["correction_error"] = {}
    #             data["correction_error"][correction.name] = None if correction_error is None else correction_error.name
    #         if changes_bad_mod:
    #             if bad_modules is None:
    #                 data.pop("bad_modules")
    #             else:
    #                 if "bad_modules" not in data.keys():
    #                     data["bad_modules"] = {}
    #                 data["bad_modules"][correction.name] = {
    #                     "count": bad_modules.count,
    #                     "relative": bad_modules.relative
    #                 }
    #         self.save_raw_data(data)
    #
    #     return changes_correction or changes_bad_mod


def read_original_matrix(original_path: Path) -> np.ndarray:
    original_path = imageio.imread(original_path)
    gray_image = color.rgb2gray(original_path)
    threshold = filters.threshold_otsu(gray_image)
    return img_as_ubyte(color.gray2rgb(gray_image > threshold))


def parse_original_qrs(labeled_images_dir: Path) -> Dict[str, np.ndarray]:
    return {
        f"{p.stem}_{f.stem}": read_original_matrix(f)
        for p in labeled_images_dir.iterdir()
        if p.is_dir()
        for f in (p / "bitmaps").iterdir()
        if f.is_file() and f.suffix == ".png"
    }


def get_original_qr(original_qrs, labeled_image: LabeledImage) -> np.ndarray:
    return original_qrs[f"{labeled_image.dataset}_{labeled_image.bitmap_id}"]


def parse_labeled_images(labeled_images_dir: Path,
                         filter_func: Optional[Callable[[LabeledImage], bool]] = None) -> List[LabeledImage]:
    return list(filter(filter_func if filter_func is not None else lambda _: True, (
        LabeledImage.from_path(image)
        for dataset in labeled_images_dir.iterdir()
        for image in (dataset / "images").iterdir()
    )))
