import os
import sys
import glob
import cv2
from ultralytics import YOLO

# ==========================================
# 1. CAU HINH CO BAN
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
MODEL_PATH = os.path.join(BASE_DIR, "runs", "vehicle_yolov8n", "weights", "best.pt")
CUSTOM_TRACKER = os.path.join(BASE_DIR, "custom_bytetrack.yaml")
TRACKER_PATH = CUSTOM_TRACKER if os.path.isfile(CUSTOM_TRACKER) else "bytetrack.yaml"

LINE_POSITION = 0.5

# --- CHE DO OUTPUT MUC THAP (TIET KIEM TAI NGUYEN & DUNG LUONG) ---
LOW_OUTPUT_MODE = True      # Bat che do nhe: xuat video nhe, chay nhanh muot, tiet kiem dung luong
TARGET_WIDTH = 640          # Chieu rong 640px (chuan YOLO), nhe gap ~9 lan Full HD
FRAME_SKIP = 2              # Lay 1 frame moi 2 frame (video goc 60 FPS -> con 30 FPS, giam tai 50%)
MAX_FRAMES = None           # None de chay tron ven toan bo video (hoac dat 250 de demo nhanh)
SAVE_SUMMARY_IMAGE = False  # False: Khong luu anh chup tom tat sau khi chay

# ID class theo dataset cua nhom
TARGET_CLASSES = [0, 1, 2]

CLASS_NAMES = {
    0: "Xe buyt",
    1: "O to",
    2: "Xe may"
}

COLORS = {
    0: (0, 165, 255),   # Cam
    1: (0, 255, 0),     # Xanh la
    2: (255, 200, 0)    # Vang/Xanh ngoc
}


# ==========================================
# 2. HAM XU LY CHO 1 VIDEO CU THE
# ==========================================
def process_video(video_path, output_path, summary_path, model):
    if not os.path.isfile(video_path):
        print(f"LOI: Khong tim thay video tai {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"LOI: Khong mo duoc video: {video_path}")
        return

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if LOW_OUTPUT_MODE and TARGET_WIDTH and orig_width > TARGET_WIDTH:
        scale = TARGET_WIDTH / orig_width
        width = int(TARGET_WIDTH)
        height = int(orig_height * scale)
    else:
        scale = 1.0
        width = int(orig_width)
        height = int(orig_height)

    # Dam bao kich thuoc chan (chia het cho 2) de codec video khong bi loi
    width = (width // 2) * 2
    height = (height // 2) * 2

    fps = (orig_fps / FRAME_SKIP) if (LOW_OUTPUT_MODE and FRAME_SKIP > 1) else orig_fps
    line_y = int(height * LINE_POSITION)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"LOI: Khong tao duoc video dau ra: {output_path}")
        cap.release()
        return

    counts = {"O to": 0, "Xe may": 0, "Xe buyt": 0}
    previous_positions = {}
    counted_ids = set()

    frame_number = 0
    frames_with_detections = 0
    frames_with_tracks = 0
    last_annotated_frame = None

    video_name = os.path.basename(video_path)
    print("\n" + "-" * 45)
    print(f"-> Dang xu ly video: {video_name}")
    print(f"   Do phan giai xuat: {width}x{height} | FPS: {fps:.1f} | Tong frame goc: {total_frames}")
    print("-" * 45)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1

            if LOW_OUTPUT_MODE and FRAME_SKIP > 1 and (frame_number % FRAME_SKIP != 0):
                continue

            if LOW_OUTPUT_MODE and MAX_FRAMES and frame_number > MAX_FRAMES:
                print(f"   [!] Da dat gioi han {MAX_FRAMES} frame demo.")
                break

            if width != orig_width or height != orig_height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            # Nhan dien & tracking
            result = model.track(
                frame,
                persist=True,
                tracker=TRACKER_PATH,
                classes=TARGET_CLASSES,
                conf=0.25,
                verbose=False
            )[0]

            annotated = frame.copy()

            # Ve vach dem
            cv2.line(annotated, (0, line_y), (width, line_y), (0, 0, 255), 3)
            cv2.putText(
                annotated,
                "COUNTING LINE",
                (20, max(30, line_y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7 if width <= 640 else 0.8,
                (0, 0, 255),
                2
            )

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                frames_with_detections += 1
                xyxy = boxes.xyxy.cpu().numpy()
                class_ids = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()

                if boxes.id is not None:
                    track_ids = boxes.id.int().cpu().tolist()
                    frames_with_tracks += 1
                else:
                    track_ids = [None] * len(class_ids)

                for box, class_id, confidence, track_id in zip(xyxy, class_ids, confidences, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = CLASS_NAMES.get(class_id, "Khac")
                    color = COLORS.get(class_id, (255, 255, 255))

                    # Khung nhan dien & tam phuong tien
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    cv2.circle(annotated, (center_x, center_y), 4, color, -1)

                    # Kiem tra qua vach dem
                    if track_id is not None:
                        track_id = int(track_id)
                        if track_id in previous_positions:
                            prev_y = previous_positions[track_id]
                            crossed = (prev_y < line_y <= center_y) or (prev_y > line_y >= center_y)
                            if crossed and track_id not in counted_ids:
                                counts[class_name] += 1
                                counted_ids.add(track_id)
                                print(f"   [+] Da dem ID {track_id}: {class_name} | Tong: {sum(counts.values())} xe")

                        previous_positions[track_id] = center_y
                        label = f"ID {track_id} | {class_name} | {confidence:.2f}"
                    else:
                        label = f"{class_name} | {confidence:.2f}"

                    label_y = max(20, y1 - 8)
                    cv2.putText(
                        annotated,
                        label,
                        (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 if width <= 640 else 0.55,
                        color,
                        2
                    )

            # Bang thong ke HUD
            panel_w = min(260, width - 20)
            panel_h = min(150, height - 20)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (10, 10), (panel_w, panel_h), (25, 25, 25), -1)
            cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

            lines = [
                "THONG KE PHUONG TIEN",
                f"O to: {counts['O to']}",
                f"Xe may: {counts['Xe may']}",
                f"Xe buyt: {counts['Xe buyt']}",
                f"TONG: {sum(counts.values())}"
            ]
            for i, text in enumerate(lines):
                font_scale = 0.5 if width <= 640 else 0.65
                spacing = 22 if width <= 640 else 28
                cv2.putText(
                    annotated,
                    text,
                    (20, 32 + i * spacing),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1 if width <= 640 and i > 0 else 2
                )

            out.write(annotated)
            last_annotated_frame = annotated

            if frame_number % 100 == 0:
                if total_frames > 0:
                    percent = frame_number * 100 / total_frames
                    print(f"   ... Tien do: {percent:.1f}% ({frame_number}/{total_frames} frame)")

    finally:
        if SAVE_SUMMARY_IMAGE and last_annotated_frame is not None and summary_path:
            cv2.imwrite(summary_path, last_annotated_frame)
        cap.release()
        out.release()

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.isfile(output_path) else 0

    print("\n" + "=" * 45)
    print(f"KET QUA DEM PHUONG TIEN [{video_name}]")
    print("=" * 45)
    for name, count in counts.items():
        print(f"  {name}: {count}")
    print(f"  TONG SO PHUONG TIEN: {sum(counts.values())}")
    print(f"  Tong frame da xu ly: {frame_number}")
    print(f"  Video ket qua: {output_path} ({file_size_mb:.2f} MB)")
    print("=" * 45)


# ==========================================
# 3. CHUONG TRINH CHINH (MENU CHON VIDEO)
# ==========================================
def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"LOI: Khong tim thay mo hinh tai {MODEL_PATH}")
        return

    # Quet danh sach cac video trong thu muc videos/
    if not os.path.isdir(VIDEOS_DIR):
        print(f"LOI: Thu muc videos khong ton tai tai {VIDEOS_DIR}")
        return

    video_files = sorted([
        f for f in os.listdir(VIDEOS_DIR)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ])

    if not video_files:
        print(f"LOI: Khong tim thay video nao trong {VIDEOS_DIR}")
        return

    print("Dang nap mo hinh YOLOv8...")
    model = YOLO(MODEL_PATH)

    # Kiem tra xem co tham so dong lenh hay khong (VD: python counter.py v02.mp4 hoac python counter.py all)
    selected_videos = []

    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg.lower() in ("all", "-a", "--all", "a"):
            selected_videos = [os.path.join(VIDEOS_DIR, vf) for vf in video_files]
        else:
            # Thu tim theo ten file hoac chi so
            if arg.isdigit() and 1 <= int(arg) <= len(video_files):
                selected_videos = [os.path.join(VIDEOS_DIR, video_files[int(arg) - 1])]
            else:
                matched = [vf for vf in video_files if arg.lower() in vf.lower()]
                if matched:
                    selected_videos = [os.path.join(VIDEOS_DIR, matched[0])]
                else:
                    print(f"Khong tim thay video khop voi '{arg}'. Chuyen ve menu lua chon.")

    # Neu chua chon qua dong lenh thi hien thi Menu de nguoi dung/thay co chon
    if not selected_videos:
        print("\n" + "=" * 55)
        print("   HE THONG NHAN DIEN VA DEM PHUONG TIEN GIAO THONG")
        print("=" * 55)
        print("Danh sach video tim thay trong thu muc 'videos/':")
        for idx, vf in enumerate(video_files, 1):
            fpath = os.path.join(VIDEOS_DIR, vf)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"   [{idx}] {vf} ({size_mb:.1f} MB)")
        print(f"   [A] Chay tat ca {len(video_files)} video tren (Batch Processing)")
        print("-" * 55)

        try:
            choice = input(f"Chon video muon chay (1-{len(video_files)} hoac A) [Mac dinh: 1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = "1"

        if not choice:
            choice = "1"

        if choice.upper() == "A":
            selected_videos = [os.path.join(VIDEOS_DIR, vf) for vf in video_files]
        elif choice.isdigit() and 1 <= int(choice) <= len(video_files):
            selected_videos = [os.path.join(VIDEOS_DIR, video_files[int(choice) - 1])]
        else:
            print(f"Lua chon '{choice}' khong hop le. Mac dinh chon video dau tien.")
            selected_videos = [os.path.join(VIDEOS_DIR, video_files[0])]

    # Tien hanh xu ly tung video da chon
    for vpath in selected_videos:
        base_name = os.path.splitext(os.path.basename(vpath))[0]
        out_video = os.path.join(BASE_DIR, f"output_{base_name}.mp4")
        out_summary = os.path.join(BASE_DIR, f"output_{base_name}_summary.jpg")

        process_video(vpath, out_video, out_summary, model)

    print("\n[V] Hoan tat toan bo qua trinh xu ly!")


if __name__ == "__main__":
    main()
