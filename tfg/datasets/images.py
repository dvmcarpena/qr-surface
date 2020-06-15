from dataclasses import dataclass
from enum import auto, Enum, unique
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import imageio
import numpy as np

from tfg.qrsurface import BadModules, Correction, QRErrorId


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
    zbar: Optional[bool]
    zbar_error: Optional[QRErrorId]
    qrs: List[LabeledQRCode]

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

        zbar = image_data.get("zbar", None)

        zbar_error_input: Optional[str] = image_data.get("zbar_error", None)
        zbar_error: Optional[QRErrorId]
        if zbar_error_input is not None:
            zbar_error = QRErrorId.__members__[zbar_error_input]
        else:
            zbar_error = None

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

        return cls(
            path=path,
            data_path=data_path,
            dataset=dataset,
            bitmap_id=bitmap_id,
            image_id=image_id,
            version=int(bitmap_data["version"]),
            has_data=(dataset_dir / "bitmaps" / f"{bitmap_id}.png").exists(),
            error_correction=bitmap_data.get("error_correction", None),
            data=bitmap_data.get("data", None),
            num_qrs=num_qrs,
            localization_error=localization_error,
            zbar=zbar,
            zbar_error=zbar_error,
            qrs=qrs,
        )

    @classmethod
    def search_labeled_images(cls, labeled_images_dir: Path,
                              filter_func: Optional[Callable[['LabeledImage'], bool]] = None) -> List['LabeledImage']:
        return list(filter(filter_func if filter_func is not None else lambda _: True, (
            cls.from_path(image)
            for dataset in labeled_images_dir.iterdir()
            for image in (dataset / "images").iterdir()
        )))

    def read_image(self) -> np.ndarray:
        return np.array(imageio.imread(str(self.path)))

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

    def update_localization_error(self, localization_error: QRErrorId, diff: Optional[Dict], update: bool = False):
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

            if diff is not None:
                from_key = str(self.localization_error)
                if from_key not in diff["localization_error"].keys():
                    diff["localization_error"][from_key] = {}

                to_key = localization_error.name
                if to_key not in diff["localization_error"][from_key].keys():
                    diff["localization_error"][from_key][to_key] = 0

                diff["localization_error"][from_key][to_key] += 1

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

    def update_successful_correction(self, correction: Correction, qr_index: int, bad_modules: BadModules,
                                     diff: Optional[Dict], update: bool = False) -> bool:
        return self._update_correction(correction, qr_index, bad_modules, None, diff, update=update)

    def update_correction_error(self, correction: Correction, qr_index: int, bad_modules: BadModules,
                                correction_error: Optional[QRErrorId], diff: Optional[Dict],
                                update: bool = False) -> bool:
        return self._update_correction(correction, qr_index, bad_modules, correction_error, diff, update=update)

    def _update_correction(self, correction: Correction, qr_index: int, bad_modules: BadModules,
                           correction_error: Optional[QRErrorId], diff: Optional[Dict], update: bool) -> bool:
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
            print(f"Correction error from {correction.name} with QR number {qr_index}: "
                  f"{source_correction} -> {dest_correction}")

            if diff is not None:
                diff_key = "correction_error"

                from_key = f"{source_correction}"
                if from_key not in diff[diff_key][correction.name].keys():
                    diff[diff_key][correction.name][from_key] = {}

                to_key = f"{dest_correction}"
                if to_key not in diff[diff_key][correction.name][from_key].keys():
                    diff[diff_key][correction.name][from_key][to_key] = 0

                diff[diff_key][correction.name][from_key][to_key] += 1
        if changes_bad_mod:
            print(f"Bad modules from {correction.name} with QR number {qr_index}: {source_bad_mod} -> {bad_modules}")

            if diff is not None:
                diff_key = "bad_modules"

                if source_bad_mod is None:
                    val_key = True
                elif bad_modules is None:
                    val_key = False
                else:
                    val_key = source_bad_mod.count > bad_modules.count

                val_key = "Better" if val_key else "Worse"
                if val_key not in diff[diff_key][correction.name].keys():
                    diff[diff_key][correction.name][val_key] = 0

                diff[diff_key][correction.name][val_key] += 1

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

    def update_zbar(self, zbar: bool, zbar_err: Optional[QRErrorId], diff: Optional[Dict], update: bool = False):
        if self.zbar is None:
            changes = True
        elif self.zbar != zbar:
            changes = True
        elif self.zbar_error != zbar_err:
            changes = True
        else:
            changes = False

        if changes:
            print()
            print(f"Zbar error: {self.zbar} {self.zbar_error} -> {zbar} {zbar_err}")

            if update:
                data = self.load_raw_data()
                data["zbar"] = zbar
                if zbar:
                    data["zbar_error"] = None if zbar_err is None else zbar_err.name
                self.save_raw_data(data)

            if diff is not None:
                diff_key = "zbar_err"
                from_key = f"{self.zbar} {self.zbar_error}"
                if from_key not in diff[diff_key].keys():
                    diff[diff_key][from_key] = {}

                to_key = f"{zbar} {zbar_err}"
                if to_key not in diff[diff_key][from_key].keys():
                    diff[diff_key][from_key][to_key] = 0

                diff[diff_key][from_key][to_key] += 1
