import cv2
from pathlib import Path

VIDEOS_DIR = "videos_for_frames"
OUTPUT_DIR = "dataset_images"

Path(OUTPUT_DIR).mkdir(exist_ok=True)

for video_path in Path(VIDEOS_DIR).glob("*.mp4"):
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_count = 0
    saved_count = 0

    print(f"обработка {video_path.name}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # cохраняем 1 кадр каждую секунду (чтобы не было кучи одинаковых фото)
        if frame_count % fps == 0:
            out_name = f"{video_path.stem}_{saved_count}.jpg"
            cv2.imwrite(str(Path(OUTPUT_DIR) / out_name), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
print(f"готово фото сохранены в папку {OUTPUT_DIR}")