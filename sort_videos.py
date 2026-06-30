import shutil
from pathlib import Path

SOURCE_FOLDER = "2026_05_23"


def organize_videos():
    source_dir = Path(SOURCE_FOLDER)

    if not source_dir.exists():
        print(f"[ОШИБКА] Папка '{SOURCE_FOLDER}' не найдена!")
        return
    folders = ["front", "middle", "rear"]
    for folder in folders:
        (source_dir / folder).mkdir(exist_ok=True)

    moved_count = 0
    skipped_count = 0

    print("Начинаю сортировку...")

    for video_path in source_dir.glob("*.mp4"):
        filename = video_path.name

        if filename.startswith("front_"):
            target_folder = "front"
        elif filename.startswith("middle_"):
            target_folder = "middle"
        elif filename.startswith("rear_"):
            target_folder = "rear"
        else:
            skipped_count += 1
            continue  # Пропускаем файлы с непонятными названиями

        target_path = source_dir / target_folder / filename
        shutil.move(str(video_path), str(target_path))
        moved_count += 1

    print("=" * 30)
    print("СОРТИРОВКА ЗАВЕРШЕНА!")
    print(f"Успешно перемещено: {moved_count} файлов.")
    if skipped_count > 0:
        print(f"Пропущено (неизвестное имя): {skipped_count} файлов.")
    print("=" * 30)


if __name__ == "__main__":
    organize_videos()