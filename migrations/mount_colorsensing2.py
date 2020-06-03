import json
from pathlib import Path

Path(f"annotations").mkdir(exist_ok=True, parents=True)
bitmap_id = "1"
for deformation in Path(bitmap_id).iterdir():
    for num_qrs in deformation.iterdir():
        for image in num_qrs.iterdir():
            qr_dict = {
                "id": image.stem,
                "id_original": bitmap_id,
                "deformation": deformation.stem,
                "num_qrs": int(num_qrs.stem)
            }
            Path(f"annotations/{image.stem}.json").write_text(json.dumps(qr_dict))
            destination = Path(f"images/{image.name}")
            if not destination.exists():
                image.replace(destination)
            else:
                raise ValueError("owerwriting stoped")

