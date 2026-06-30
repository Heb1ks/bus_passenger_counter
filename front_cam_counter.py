import argparse
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import platform

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# модель
HEAD_MODEL_LOCAL = "best.pt"

# геометрия двери
DOOR_X_LEFT_RATIO  = 0.20
DOOR_X_RIGHT_RATIO = 0.79

STREET_Y_LIMIT_RATIO = 0.25   # стрит бокс нижняя граница
STREET_ENTRY_Y_RATIO = 0.70
STREET_RED_LINE_RATIO = 0.05  # красная линия
DOOR_Y_LIMIT_RATIO   = 0.45
IN_START_TOP_RATIO = 0.35

VECTOR_IN_DY_RATIO  = 0.08
VECTOR_OUT_DY_RATIO = 0.12

TRACKING_POINT_Y_OFFSET = 0.38
POST_IN_REDLINE_FRAMES = 25

# детекция и трекинг конфиг
CONF   = 0.25
IOU    = 0.35
IMGSZ  = 640

MIN_BOX_AREA_RATIO = 0.001
MAX_BOX_AREA_RATIO = 0.25
MIN_ASPECT = 0.35
MAX_ASPECT = 2.8
EDGE_MARGIN_X_RATIO = 0.05

MIN_TRACK_FRAMES   = 3
MIN_TRACK_DIST_RATIO = 0.01
COOLDOWN_FRAMES    = 60
GHOST_MAX_GAP      = 20
GHOST_RADIUS_RATIO = 0.15
MERGE_RADIUS_RATIO = 0.10
MAX_BAG_RATIO      = 0.30
IOA_THRESH         = 0.50

TRACKER_CFG = {
    "tracker_type": "botsort",
    "track_high_thresh": 0.30,
    "track_low_thresh":  0.05,
    "new_track_thresh":  0.30,
    "track_buffer":      60,
    "match_thresh":      0.85,
    "fuse_score":        True,
    "gmc_method":        "sparseOptFlow",
    "proximity_thresh":  0.5,
    "appearance_thresh": 0.25,
    "with_reid":         False,
    "model":             None,
}

VIDEOS_DIR  = "2026_05_22/front"
DEFAULT_VID = "2026_05_22/front/front_2026-05-22_06-02-52_seg001.mp4"
TARGET_WIDTH = 1024


# ghost трекер
class GhostTracker:
    def __init__(self, radius: float, max_gap: int = GHOST_MAX_GAP, min_alive: int = 2):
        self.radius    = radius
        self.max_gap   = max_gap
        self.min_alive = min_alive
        self._ghosts:   dict[int, dict] = {}
        self._restored: dict[int, int]  = {}

    def update_lost(self, tid: int, cx: float, cy: float, frame_idx: int, alive: int):
        if alive >= self.min_alive:
            self._ghosts[tid] = {"cx": cx, "cy": cy, "frame": frame_idx}

    def try_restore(self, new_tid: int, cx: float, cy: float, frame_idx: int, counted_out_ids: set) -> int | None:
        if new_tid in self._restored:
            return self._restored[new_tid]
        best_dist, best_old = self.radius + 1, None
        for old_tid, g in self._ghosts.items():
            if old_tid in counted_out_ids:  # не востонавливает out треки
                continue
            if frame_idx - g["frame"] > self.max_gap:
                continue
            d = np.hypot(cx - g["cx"], cy - g["cy"])
            if d < best_dist:
                best_dist, best_old = d, old_tid
        if best_old is not None:
            self._restored[new_tid] = best_old
            del self._ghosts[best_old]
            return best_old
        return None

    def cleanup(self, frame_idx: int):
        expired = [k for k, v in self._ghosts.items()
                   if frame_idx - v["frame"] > self.max_gap]
        for k in expired:
            self._ghosts.pop(k, None)


def load_model(local_path: str = HEAD_MODEL_LOCAL) -> YOLO:
    p = Path(local_path)
    if p.exists():
        print(f"[MODEL] Загружена: {p}")
        model = YOLO(str(p))
        model.to("cuda")
        return model
    print(f"[ОШИБКА] Файл модели '{local_path}' не найден!")
    exit(1)


# оверлей
def draw_overlay(frame, fw, fh, count_in, count_out, frame_idx):
    door_y = int(DOOR_Y_LIMIT_RATIO * fh)

    sx1 = int(DOOR_X_LEFT_RATIO * fw)
    sx2 = int(DOOR_X_RIGHT_RATIO * fw)
    sy  = int(STREET_Y_LIMIT_RATIO * fh)
    red_y = int(STREET_RED_LINE_RATIO * fh)

    # синия коробка стрит бокс
    cv2.rectangle(frame, (sx1, 0), (sx2, sy), (255, 100, 0), 2)
    cv2.putText(frame, "STREET BOX", (sx1 + 5, sy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

    # красная линия
    cv2.line(frame, (sx1, red_y), (sx2, red_y), (0, 0, 255), 2)
    cv2.putText(frame, "EXIT LINE", (sx1 + 5, red_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # жёлтая граница
    cv2.line(frame, (0, door_y), (fw, door_y), (0, 200, 255), 2)
    cv2.putText(frame, "DOOR LIMIT", (6, door_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    cv2.putText(frame, f"IN : {count_in}",  (16, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 230, 0),   2)
    cv2.putText(frame, f"OUT: {count_out}", (16, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 100, 230), 2)
    cv2.putText(frame, f"f={frame_idx}", (fw - 120, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


# основа
def process_video(source: str, model: YOLO, tracker_path: str,
                  show: bool, save_debug: bool) -> dict:

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ОШИБКА] Не удалось открыть: {source}")
        return {}

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0

    if fw > TARGET_WIDTH:
        scale = TARGET_WIDTH / fw
        fw = TARGET_WIDTH
        fh = int(fh * scale)

    edge_x       = int(fw * EDGE_MARGIN_X_RATIO)
    track_dist   = int(fh * MIN_TRACK_DIST_RATIO)
    ghost_radius = int(fw * GHOST_RADIUS_RATIO)
    merge_radius = int(fw * MERGE_RADIUS_RATIO)
    frame_area   = fw * fh
    min_box_area = frame_area * MIN_BOX_AREA_RATIO
    max_box_area = frame_area * MAX_BOX_AREA_RATIO

    # пиксельные пороги геометрии
    x_min   = fw * DOOR_X_LEFT_RATIO
    x_max   = fw * DOOR_X_RIGHT_RATIO
    y_street = fh * STREET_Y_LIMIT_RATIO
    y_red    = fh * STREET_RED_LINE_RATIO   # красная линия

    street_entry_threshold = y_street * STREET_ENTRY_Y_RATIO

    dummy = np.zeros((fh, fw, 3), dtype=np.uint8)
    for _ in range(3):
        model.predict(dummy, verbose=False, imgsz=IMGSZ, device="cuda")

    writer = None
    if save_debug:
        out_path = str(Path(source).stem) + "_debug_front.mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

    track_alive:        dict[int, int]                 = defaultdict(int)
    last_known_pos:     dict[int, tuple[float, float]] = {}
    first_seen_frame:   dict[int, int]                 = {}
    first_seen_pos:     dict[int, tuple[float, float]] = {}
    last_counted_frame: dict[int, int]                 = defaultdict(lambda: -(COOLDOWN_FRAMES + 1))
    anchor_pos:         dict[int, tuple[float, float]] = {}

    counted_in_ids:  set[int] = set()
    counted_out_ids: set[int] = set()

    # для каждого трека стартовал ли он в стрит бокс первого фрейма
    started_in_street: set[int] = set()
    # трекам которые прошли красную линию
    crossed_red_line:  set[int] = set()
    crossed_red_line_in: set[int] = set()
    # фрейм  когда трек был засчитан как IN
    in_counted_frame:  dict[int, int] = {}

    ghost   = GhostTracker(radius=ghost_radius)
    tid_map: dict[int, int] = {}

    count_in = count_out = 0
    events   = []
    frame_idx = 0
    prev_real_tids: set[int] = set()
    show_window = show

    if show_window:
        cv2.namedWindow("Front Cam", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Front Cam", 1280, 720)

    def log_event(action: str, tid: int):
        nonlocal count_in, count_out
        ts = datetime.now().strftime("%H:%M:%S")
        events.append((ts, action, tid))
        print(f"  [{ts}] {action}  id={tid}  IN={count_in} OUT={count_out}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame.shape[1] > TARGET_WIDTH:
            frame = cv2.resize(frame, (fw, fh))

        results = model.track(
            source=frame, classes=[0],
            conf=CONF, iou=IOU, imgsz=IMGSZ,
            tracker=tracker_path, persist=True, verbose=False, augment=True , half=True,
            device = "cuda"
        )

        sized = []
        if (results and results[0].boxes is not None
                and results[0].boxes.id is not None):
            for xyxy, tid_raw in zip(
                results[0].boxes.xyxy.cpu().numpy(),
                results[0].boxes.id.cpu().numpy().astype(int)
            ):
                x1, y1, x2, y2 = xyxy
                w_box = x2 - x1; h_box = y2 - y1
                area   = w_box * h_box
                aspect = w_box / max(h_box, 1)
                cx = (x1 + x2) / 2.0
                cy = y2 - (y2 - y1) * TRACKING_POINT_Y_OFFSET

                if cx < edge_x or cx > (fw - edge_x): continue
                if (area < min_box_area or area > max_box_area
                        or not (MIN_ASPECT <= aspect <= MAX_ASPECT)): continue

                if tid_raw not in tid_map:
                    old = ghost.try_restore(tid_raw, cx, cy, frame_idx, counted_out_ids)
                    tid_map[tid_raw] = old if old is not None else tid_raw

                canonical = tid_map[tid_raw]
                sized.append((xyxy, canonical, cx, cy, area))

        lost_tids = prev_real_tids - {s[1] for s in sized}
        for lt in lost_tids:
            if lt in last_known_pos:
                lx, ly = last_known_pos[lt]
                ghost.update_lost(lt, lx, ly, frame_idx, track_alive[lt])
        ghost.cleanup(frame_idx)

        suppressed: set[int] = set()
        for i in range(len(sized)):
            if sized[i][1] in suppressed: continue
            cx_i, cy_i, area_i = sized[i][2], sized[i][3], sized[i][4]
            ax1, ay1, ax2, ay2 = sized[i][0]
            for j in range(i + 1, len(sized)):
                if sized[j][1] in suppressed: continue
                cx_j, cy_j, area_j = sized[j][2], sized[j][3], sized[j][4]
                bx1, by1, bx2, by2 = sized[j][0]
                dist  = np.hypot(cx_i - cx_j, cy_i - cy_j)
                inter = (max(0, min(ax2, bx2) - max(ax1, bx1)) *
                         max(0, min(ay2, by2) - max(ay1, by1)))
                min_a = min(area_i, area_j)
                max_a = max(area_i, area_j)
                ioa   = inter / min_a if min_a > 0 else 0
                if ioa > IOA_THRESH or (dist < merge_radius
                                        and min_a / max_a < MAX_BAG_RATIO):
                    suppressed.add(sized[j][1] if area_i >= area_j else sized[i][1])
                    if area_i < area_j: break

        for xyxy, tid, cx, cy, area in sized:
            if tid in suppressed: continue

            x1, y1, x2, y2 = xyxy
            track_alive[tid] += 1
            last_known_pos[tid] = (cx, cy)

            if tid not in first_seen_frame:
                first_seen_frame[tid] = frame_idx
                first_seen_pos[tid]   = (cx, cy)
                # запоминает страртовал ли трек в стрит бокс
                if (x_min < cx < x_max) and (cy < street_entry_threshold):
                    started_in_street.add(tid)

            frames_alive = frame_idx - first_seen_frame[tid]
            sx, sy_start = first_seen_pos[tid]
            total_disp = np.hypot(cx - sx, cy - sy_start)
            is_valid   = (frames_alive >= MIN_TRACK_FRAMES) and (total_disp >= track_dist)
            cooldown_ok = (frame_idx - last_counted_frame[tid]) > COOLDOWN_FRAMES

            # обновляем anchor
            if tid not in anchor_pos:
                anchor_pos[tid] = (cx, cy)

            dy = cy - anchor_pos[tid][1]

            IN_THRESH  = fh * VECTOR_IN_DY_RATIO
            OUT_THRESH = fh * VECTOR_OUT_DY_RATIO

            start_x_pos, start_cy = first_seen_pos[tid]

            started_on_street = tid in started_in_street
            currently_on_street = (x_min < cx < x_max) and (cy < y_street)

            top_entry_limit = y_red + fh * IN_START_TOP_RATIO

            came_from_top = start_cy < top_entry_limit

            total_dy_from_start = cy - first_seen_pos[tid][1]

            is_moving_in = (
                    dy > IN_THRESH
                    and started_on_street
                    and came_from_top
                    and total_dy_from_start > fh * 0.10
            )

            total_dy_from_start_out = first_seen_pos[tid][1] - cy
            started_below_street = first_seen_pos[tid][1] > y_street

            is_moving_out = (
                    dy < -OUT_THRESH
                    and currently_on_street
                    and total_dy_from_start_out > fh * 0.10
                    and not started_below_street
            )

            is_moving_out_from_inside = (
                    started_below_street
                    and currently_on_street
                    and total_dy_from_start_out > fh * 0.25  # прошёл 25% кадра вверх
                    and dy < -OUT_THRESH
                    and (tid not in counted_out_ids)
                    and (tid not in counted_in_ids)
            )

            # if dy > IN_THRESH * 0.5 and tid not in counted_in_ids:
            #     print(f"  [BLOCKED-IN] tid={tid} f={frame_idx} "
            #           f"dy={dy:.1f}/{IN_THRESH:.1f} "
            #           f"valid={is_valid} cooldown={cooldown_ok} "
            #           f"street={started_on_street} top={came_from_top}")


            if tid in counted_in_ids:
                is_moving_in = False

                # стандартный IN
            if is_moving_in and is_valid and cooldown_ok and (tid not in counted_in_ids):
                    count_in += 1
                    counted_in_ids.add(tid)
                    counted_out_ids.discard(tid)
                    anchor_pos[tid] = (cx, cy)
                    last_counted_frame[tid] = frame_idx
                    in_counted_frame[tid] = frame_idx
                    log_event("IN", tid)
                    cv2.circle(frame, (int(cx), int(cy)), 18, (0, 255, 0), -1)

                    # fallback IN через красную линию - сверху в низ
            elif (tid not in counted_in_ids
                          and tid not in crossed_red_line_in
                          and cy > y_red
                          and dy > IN_THRESH * 0.5
                          and in_x_zone
                          and is_valid
                          and started_on_street  #  обязательно трек стартовал на улице
                          and came_from_top  #  первая позиция была выше red_y + IN_START_TOP
                          and frames_alive < 90  #  трек молодой не стоял давно в кадре
                          and first_cy < y_street):  #  первая позиция была в зоне улицы
                    crossed_red_line_in.add(tid)
                    count_in += 1
                    counted_in_ids.add(tid)
                    counted_out_ids.discard(tid)
                    anchor_pos[tid] = (cx, cy)
                    last_counted_frame[tid] = frame_idx
                    in_counted_frame[tid] = frame_idx
                    log_event("IN(redline)", tid)
                    cv2.circle(frame, (int(cx), int(cy)), 22, (0, 255, 0), -1)
                    cv2.putText(frame, "IN!", (int(cx) + 10, int(cy)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # стандартный OUT
            elif (is_moving_out or is_moving_out_from_inside) and is_valid and cooldown_ok and (tid not in counted_out_ids):
                # После IN - стандартный OUT полностью заблокирован только красная линия
                if tid in counted_in_ids:
                    pass  # только через красную линию
                else:
                    count_out += 1
                    counted_out_ids.add(tid)
                    counted_in_ids.discard(tid)
                    anchor_pos[tid] = (cx, cy)
                    last_counted_frame[tid] = frame_idx
                    log_event("OUT", tid)
                    cv2.circle(frame, (int(cx), int(cy)), 18, (0, 80, 255), -1)

            # fallback OUT через красную линию
            in_x_zone = (x_min < cx < x_max)
            above_red = (cy < y_red)

            _, first_cy = first_seen_pos[tid]
            anchor_cx_val, anchor_cy_val = anchor_pos.get(
                tid,
                first_seen_pos[tid]
            )

            # реально движется вверх относительно anchor
            # минимальное смещение вверх — 2% высоты кадра
            moving_upward = (cy < anchor_cy_val - fh * 0.02)

            # не успел уйти глубоко в автобус
            never_went_deep = (cy < y_street)

            # если dy > 0  значит человек движется вниз - в автобус
            is_going_down = (dy > 0)

            if ((started_on_street or tid in counted_in_ids)
                    and (tid not in counted_out_ids)
                    and (tid not in crossed_red_line)
                    and above_red
                    and in_x_zone
                    and moving_upward
                    and never_went_deep
                    and not is_going_down):
                crossed_red_line.add(tid)
                count_out += 1
                counted_out_ids.add(tid)
                counted_in_ids.discard(tid)
                last_counted_frame[tid] = frame_idx

                log_event("OUT(redline)", tid)

                cv2.circle(frame, (int(cx), int(cy)),
                           22, (0, 0, 255), -1)

                cv2.putText(frame,
                            "OUT!",
                            (int(cx) + 10, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2)

                # ── OUT для людей которые были IN ────────────────
            if (tid in counted_in_ids
                    and tid not in counted_out_ids
                    and tid not in crossed_red_line
                    and above_red
                    and in_x_zone
                    and moving_upward):
                crossed_red_line.add(tid)
                count_out += 1
                counted_out_ids.add(tid)
                counted_in_ids.discard(tid)
                last_counted_frame[tid] = frame_idx
                log_event("OUT(redline-in)", tid)
                cv2.circle(frame, (int(cx), int(cy)), 22, (0, 0, 255), -1)
                cv2.putText(frame, "OUT!", (int(cx) + 10, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if tid in counted_in_ids and cy > anchor_pos[tid][1]:
                anchor_pos[tid] = (cx, cy)
            elif tid in counted_out_ids and cy < anchor_pos[tid][1]:
                anchor_pos[tid] = (cx, cy)

            is_counted = (tid in counted_in_ids) or (tid in counted_out_ids)
            if not is_valid:   box_color = (0, 0, 255)
            elif is_counted:   box_color = (0, 200, 200)
            else:              box_color = (200, 200, 0)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"#{tid}", (int(x1), max(int(y1) - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

        prev_real_tids = {s[1] for s in sized if s[1] not in suppressed}

        draw_overlay(frame, fw, fh, count_in, count_out, frame_idx)

        if writer: writer.write(frame)
        frame_idx += 1

        if show_window:
            try:
                cv2.imshow("Front Cam", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
            except cv2.error:
                show_window = False

    cap.release()
    if writer: writer.release()
    if show_window: cv2.destroyAllWindows()

    return {"file": source, "in": count_in, "out": count_out, "events": events}


def save_log(results: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Отчёт счётчика пассажиров — ПЕРЕДНЯЯ КАМЕРА\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        total_in = total_out = 0
        for r in results:
            if not r: continue
            f.write(f"Файл: {Path(r['file']).name}\n")
            f.write(f"  Вошли  (IN):  {r['in']}\n")
            f.write(f"  Вышли (OUT): {r['out']}\n")
            total_in  += r["in"]
            total_out += r["out"]
        f.write("=" * 50 + "\n")
        f.write(f"ИТОГО  IN={total_in}  OUT={total_out}\n")
    print(f"\n[LOG] Сохранён: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",     default=DEFAULT_VID)
    ap.add_argument("--all",        action="store_true")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--save-debug", action="store_true")
    ap.add_argument("--model",      default=HEAD_MODEL_LOCAL)
    ap.add_argument("--log",        default="front_log.txt")
    args = ap.parse_args()

    model = load_model(args.model)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                     delete=False, encoding="utf-8")
    yaml.dump(TRACKER_CFG, tmp)
    tmp.close()

    try:
        if args.all:
            videos  = sorted(Path(VIDEOS_DIR).glob("*.mp4"))
            results = [process_video(str(v), model, tmp.name,
                                     not args.no_display, args.save_debug)
                       for v in videos]
        else:
            results = [process_video(args.source, model, tmp.name,
                                     not args.no_display, args.save_debug)]
    finally:
        Path(tmp.name).unlink(missing_ok=True)




    save_log(results, args.log)



    system = platform.system()
    if system == "Darwin":  # мак
        import os
        os.system("afplay /System/Library/Sounds/Glass.aiff")
    elif system == "Windows":
        import winsound
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    elif system == "Linux":
        import os
        os.system("paplay /usr/share/sounds/freedesktop/stereo/complete.oga 2>/dev/null "
                  "|| aplay /usr/share/sounds/alsa/Front_Center.wav 2>/dev/null")


if __name__ == "__main__":
    main()