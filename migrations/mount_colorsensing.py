import json
from pathlib import Path

bitmap_id = "1"
for deformation in Path(bitmap_id).iterdir():
    for num_qrs in deformation.iterdir():
        for p3 in num_qrs.iterdir():
            if p3.stem == "bad":
                for p4 in p3.iterdir():
                    persperctive_result = p4.stem

                    for image in p4.iterdir():
                        qr_dict = {
                            "id": image.stem,
                            "id_original": bitmap_id,
                            "deformation": deformation.stem,
                            "num_qrs": int(num_qrs.stem),
                            "corrections": {
                                "projective": persperctive_result
                            }
                        }
                        Path(f"annotations/{image.stem}.json").write_text(json.dumps(qr_dict))
                        destination = Path(f"images/{image.name}")
                        if not destination.exists():
                            image.replace(destination)
                        else:
                            raise ValueError("owerwriting stoped")
            else:
                persperctive_result = p3.stem
                
                for image in p3.iterdir():
                    qr_dict = {
                        "id": image.stem,
                        "id_original": bitmap_id,
                        "deformation": deformation.stem,
                        "num_qrs": int(num_qrs.stem),
                        "corrections": {
                            "projective": persperctive_result
                        }
                    }
                    Path(f"annotations/{image.stem}.json").write_text(json.dumps(qr_dict))
                    destination = Path(f"images/{image.name}")
                    if not destination.exists():
                        image.replace(destination)
                    else:
                        raise ValueError("owerwriting stoped")
