
from pathlib import Path
from ultralytics import YOLO

# 1. Xac dinh thu muc du an
BASE_DIR = Path(__file__).resolve().parent

# 2. Duong dan file cau hinh dataset
DATA_YAML = BASE_DIR / "dataset" / "data.yaml"

# 3. Duong dan mo hinh YOLO khoi tao
MODEL_PATH = BASE_DIR / "dataset" / "yolov8n.pt"

# 4. Kiem tra cac file can thiet
if not DATA_YAML.is_file():
    raise FileNotFoundError(
        f"Khong tim thay data.yaml: {DATA_YAML}"
    )

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Khong tim thay mo hinh: {MODEL_PATH}"
    )

# 5. Nap mo hinh
model = YOLO(str(MODEL_PATH))

# 6. Huan luyen
results = model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    project=str(BASE_DIR / "runs"),
    name="vehicle_yolov8n",
    exist_ok=True
)

print("Da hoan thanh lenh huan luyen.")
print("Kiem tra thu muc:", BASE_DIR / "runs" / "vehicle_yolov8n")
