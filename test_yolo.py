
from pathlib import Path
from ultralytics import YOLO

# 1. Duong dan video dau vao
video_path = Path("videos/v01.mp4")

# 2. Kiem tra video co ton tai khong
if not video_path.exists():
    raise FileNotFoundError(
        f"Khong tim thay video: {video_path.resolve()}"
    )

# 3. Tai mo hinh YOLO co san
model = YOLO("yolov8n.pt")

# 4. Nhan dien cac doi tuong trong video
model.predict(
    source=str(video_path),
    conf=0.25,
    save=True,
    project="outputs",
    name="yolo_test",
    exist_ok=True
)

print("Da chay xong. Kiem tra thu muc outputs/yolo_test")
