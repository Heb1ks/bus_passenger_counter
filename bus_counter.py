import argparse
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

HEAD_MODEL_LOCAL = "best.pt"

HEAD_MODEL_SOURCES = []


LINE_A_P1 = (0, 260)  # ВЕРХНЯЯ ЛИНИЯ (Улица)
LINE_A_P2 = (1280, 248)

LINE_B_P1 = (0, 340)  # НИЖНЯЯ ЛИНИЯ (Салон)
LINE_B_P2 = (1280, 340)

CONF = 0.15
IOU = 0.35
IMGSZ = 640

MIN_BOX_AREA = 300
MAX_BOX_AREA = 200000
MIN_ASPECT = 0.4
MAX_ASPECT = 2.5

MERGE_RADIUS = 90
MAX_BAG_RATIO = 0.4
IOA_THRESH = 0.50
EDGE_MARGIN_X = 120

MIN_TRACK_FRAMES = 4

TRACKER_CFG = {
    "tracker_type": "botsort",
    "track_high_thresh": 0.15,
    "track_low_thresh": 0.05,
    "new_track_thresh": 0.20,
    "track_buffer": 60,
    "match_thresh": 0.85,
    "fuse_score": True,
    "gmc_method": "sparseOptFlow",
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.25,
    "with_reid": False,
    "model": None,
}

COOLDOWN_FRAMES = 15

GHOST_RADIUS_PX = 140
GHOST_MAX_GAP = 40
GHOST_MIN_ALIVE = 2

PROCESS_EVERY_N = 1
DEBUG = False
DEBUG_LOG_FILE = "debug_passenger.log"

VIDEOS_DIR = "videos"
DEFAULT_VID = "videos/kakoetovideo.mp4"


def dbg(msg: str, log_fh=None):
    if DEBUG:
        print(f"[DBG] {msg}")
        if log_fh:
            log_fh.write(f"[DBG] {msg}\n")
            log_fh.flush()


def _download_url(url: str, dest: str) -> bool:
    import urllib.request
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except:
        return False


def _download_hf(repo_id: str, filename: str, dest: str) -> bool:
    try:
        from huggingface_hub import hf_hub_download
        tmp = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=".")
        Path(tmp).rename(dest)
        return True
    except:
        return False


def load_head_model(local_path: str = HEAD_MODEL_LOCAL) -> tuple[YOLO, bool]:
    p = Path(local_path)
    if p.exists(): return YOLO(str(p)), True

    for source in HEAD_MODEL_SOURCES:
        if source[0] == "hf" and _download_hf(source[1], source[2], local_path): return YOLO(local_path), True
        if source[0] == "url" and _download_url(source[1], local_path): return YOLO(local_path), True

    return YOLO("yolov8n.pt"), False


def extend_line_to_frame(p1, p2, frame_w, frame_h):
    x1, y1 = p1
    x2, y2 = p2
    if x2 == x1: return (x1, 0), (x1, frame_h)
    slope = (y2 - y1) / (x2 - x1)
    y_left = int(y1 - slope * x1)
    y_right = int(y1 + slope * (frame_w - x1))
    return (0, y_left), (frame_w, y_right)


class GhostTracker:
    def __init__(self, radius: float = GHOST_RADIUS_PX, max_gap: int = GHOST_MAX_GAP, min_alive: int = GHOST_MIN_ALIVE):
        self.radius, self.max_gap, self.min_alive = radius, max_gap, min_alive
        self._ghosts, self._restored = {}, {}

    def update_lost(self, tid: int, cx: float, cy: float, frame_idx: int, alive_frames: int):
        if alive_frames >= self.min_alive:
            self._ghosts[tid] = {"cx": cx, "cy": cy, "frame": frame_idx, "alive": alive_frames}

    def try_restore(self, new_tid: int, cx: float, cy: float, frame_idx: int) -> int | None:
        if new_tid in self._restored: return self._restored[new_tid]
        best_dist, best_old = self.radius + 1, None
        for old_tid, g in self._ghosts.items():
            if frame_idx - g["frame"] > self.max_gap: continue
            d = np.hypot(cx - g["cx"], cy - g["cy"])
            if d < best_dist: best_dist, best_old = d, old_tid
        if best_old is not None:
            self._restored[new_tid] = best_old
            del self._ghosts[best_old]
            return best_old
        return None

    def cleanup(self, frame_idx: int):
        expired = [k for k, v in self._ghosts.items() if frame_idx - v["frame"] > self.max_gap]
        for k in expired: self._ghosts.pop(k, None)


def draw_overlay(frame, ea1, ea2, eb1, eb2, ci, co, frame_idx, zone_color=(0, 60, 0)):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (EDGE_MARGIN_X, h), (0, 0, 255), -1)
    cv2.rectangle(overlay, (w - EDGE_MARGIN_X, 0), (w, h), (0, 0, 255), -1)

    pts = np.array([ea1, ea2, eb2, eb1], dtype=np.int32)
    cv2.fillPoly(overlay, [pts], zone_color)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    cv2.line(frame, ea1, ea2, (0, 80, 255), 2)
    cv2.putText(frame, "LINE A (street) [OUT]", (ea1[0] + 5, ea1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 80, 255),
                1)

    cv2.line(frame, eb1, eb2, (0, 220, 60), 2)
    cv2.putText(frame, "LINE B (bus) [IN]", (eb1[0] + 5, eb1[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 60), 1)

    cv2.putText(frame, f"IN : {ci}", (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 230, 0), 2)
    cv2.putText(frame, f"OUT: {co}", (16, 84), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 100, 230), 2)
    cv2.putText(frame, f"f={frame_idx}", (w - 120, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)


def process_video(source: str, model: YOLO, is_head_model: bool, tracker_path: str, show: bool, save_debug: bool,
                  debug_log_fh=None) -> dict:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): return {}

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0

    dummy = np.zeros((fh, fw, 3), dtype=np.uint8)
    for _ in range(3): model.predict(dummy, verbose=False, imgsz=IMGSZ)

    detect_classes = [0]
    writer = None
    if save_debug:
        out_path = str(Path(source).stem) + "_debug_final.mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

    ea1, ea2 = extend_line_to_frame(LINE_A_P1, LINE_A_P2, fw, fh)
    eb1, eb2 = extend_line_to_frame(LINE_B_P1, LINE_B_P2, fw, fh)

    ay_left, ay_right = ea1[1], ea2[1]
    by_left, by_right = eb1[1], eb2[1]

    last_counted_frame: dict[int, int] = defaultdict(lambda: -(COOLDOWN_FRAMES + 1))
    track_alive: dict[int, int] = defaultdict(int)
    last_known_pos: dict[int, tuple[float, float]] = {}

    first_seen_frame: dict[int, int] = {}
    first_seen_y: dict[int, float] = {}
    counted_status: dict[int, str] = defaultdict(lambda: None)

    ghost = GhostTracker()
    tid_map: dict[int, int] = {}

    count_in = count_out = 0
    events = []
    frame_idx = 0
    show_window = show
    prev_real_tids: set[int] = set()

    def log_event(action: str, tid: int):
        nonlocal count_in, count_out
        ts = datetime.now().strftime("%H:%M:%S")
        events.append((ts, action, tid))
        msg = f"  [{ts}] {action}  id={tid}  IN={count_in} OUT={count_out}"
        print(msg)
        if debug_log_fh:
            debug_log_fh.write(msg + "\n")
            debug_log_fh.flush()

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = model.track(
            source=frame, classes=detect_classes,
            conf=CONF, iou=IOU, imgsz=IMGSZ,
            tracker=tracker_path, persist=True, verbose=False, augment=True
        )

        real_tids: set[int] = set()
        sized = []

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            for xyxy, tid_raw in zip(results[0].boxes.xyxy.cpu().numpy(),
                                     results[0].boxes.id.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = xyxy
                w_box = x2 - x1;
                h_box = y2 - y1
                area = w_box * h_box;
                aspect = w_box / max(h_box, 1)
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                if cx < EDGE_MARGIN_X or cx > (fw - EDGE_MARGIN_X): continue
                if area < MIN_BOX_AREA or area > MAX_BOX_AREA or not (MIN_ASPECT <= aspect <= MAX_ASPECT): continue

                if tid_raw not in tid_map:
                    old_tid = ghost.try_restore(tid_raw, cx, cy, frame_idx)
                    tid_map[tid_raw] = old_tid if old_tid is not None else tid_raw

                canonical_tid = tid_map[tid_raw]
                real_tids.add(canonical_tid)
                sized.append((xyxy, canonical_tid, cx, cy, area))

        lost_tids = prev_real_tids - {s[1] for s in sized}
        for lost_tid in lost_tids:
            if lost_tid in last_known_pos:
                lx, ly = last_known_pos[lost_tid]
                ghost.update_lost(lost_tid, lx, ly, frame_idx, track_alive[lost_tid])
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

                dist = np.hypot(cx_i - cx_j, cy_i - cy_j)
                inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
                min_area = min(area_i, area_j)
                max_area = max(area_i, area_j)

                ioa = inter / min_area if min_area > 0 else 0
                is_overlap = ioa > IOA_THRESH
                is_bag_nearby = (dist < MERGE_RADIUS) and (min_area / max_area < MAX_BAG_RATIO)

                if is_overlap or is_bag_nearby:
                    if area_i >= area_j:
                        suppressed.add(sized[j][1])
                    else:
                        suppressed.add(sized[i][1])
                        break

        for xyxy, tid, cx, cy, area in sized:
            if tid in suppressed: continue

            x1, y1, x2, y2 = xyxy
            track_alive[tid] += 1
            last_known_pos[tid] = (cx, cy)

            # ТОЧКА РОЖДЕНИЯ
            if tid not in first_seen_frame:
                first_seen_frame[tid] = frame_idx
                first_seen_y[tid] = cy  # Запоминаем Y-координату первого появления!

            frames_alive = frame_idx - first_seen_frame[tid]
            is_mature_track = frames_alive >= MIN_TRACK_FRAMES
            spawn_y = first_seen_y[tid]

            t_curr = cx / max(fw - 1, 1)
            line_a_y_curr = ay_left + t_curr * (ay_right - ay_left)
            line_b_y_curr = by_left + t_curr * (by_right - by_left)


            in_bus_zone = cy >= line_b_y_curr
            spawned_above = spawn_y < (line_b_y_curr - 20)
            entered_bus = in_bus_zone and spawned_above

            in_street_zone = cy <= line_a_y_curr
            spawned_below = spawn_y > (line_a_y_curr + 20)
            entered_street = in_street_zone and spawned_below

            cooldown_ok = (frame_idx - last_counted_frame[tid]) > COOLDOWN_FRAMES

            if entered_bus and cooldown_ok and is_mature_track and counted_status[tid] != 'IN':
                counted_status[tid] = 'IN'
                last_counted_frame[tid] = frame_idx
                count_in += 1
                log_event("IN", tid)
                cv2.circle(frame, (int(cx), int(cy)), 15, (0, 255, 0), -1)
                cv2.line(frame, eb1, eb2, (0, 255, 0), 5)

            elif entered_street and cooldown_ok and is_mature_track and counted_status[tid] != 'OUT':
                counted_status[tid] = 'OUT'
                last_counted_frame[tid] = frame_idx
                count_out += 1
                log_event("OUT", tid)
                cv2.circle(frame, (int(cx), int(cy)), 15, (0, 80, 255), -1)
                cv2.line(frame, ea1, ea2, (0, 80, 255), 5)

            # Отрисовка
            if counted_status[tid] == 'IN':
                box_color = (0, 200, 200)
            elif counted_status[tid] == 'OUT':
                box_color = (200, 0, 200)
            else:
                box_color = (200, 200, 0)

            if not is_mature_track:
                box_color = (0, 0, 255)

            is_valid_spawn = spawned_above or spawned_below
            if is_mature_track and not is_valid_spawn:
                box_color = (128, 128, 128)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"#{tid}", (int(x1), max(int(y1) - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

        prev_real_tids = {s[1] for s in sized if s[1] not in suppressed}

        draw_overlay(frame, ea1, ea2, eb1, eb2, count_in, count_out, frame_idx)

        if writer: writer.write(frame)
        frame_idx += 1

        if show_window:
            try:
                cv2.imshow("Spawn Validator", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
            except cv2.error:
                show_window = False

    cap.release()
    if writer: writer.release()
    if show_window: cv2.destroyAllWindows()

    return {"file": source, "in": count_in, "out": count_out, "events": events}


def save_log(results, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Отчёт счётчика пассажиров (Spawn Validation)\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        total_in = total_out = 0
        for r in results:
            if not r: continue
            f.write(f"Файл: {Path(r['file']).name}\n")
            f.write(f"  Вошли  (IN):  {r['in']}\n")
            f.write(f"  Вышли (OUT): {r['out']}\n")
            total_in += r["in"]
            total_out += r["out"]
        f.write("=" * 50 + "\n")
        f.write(f"ИТОГО  IN={total_in}  OUT={total_out}\n")


def main():
    global DEBUG
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_VID)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--save-debug", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--model", default=HEAD_MODEL_LOCAL)
    ap.add_argument("--log", default="spawn_validation_log.txt")
    args = ap.parse_args()

    if args.debug: DEBUG = True
    debug_log_fh = open(DEBUG_LOG_FILE, "w", encoding="utf-8") if DEBUG else None

    model, is_head_model = load_head_model(args.model)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.dump(TRACKER_CFG, tmp)
    tmp.close()

    try:
        if args.all:
            videos = sorted(Path(VIDEOS_DIR).glob("*.mp4"))
            results = [process_video(str(v), model, is_head_model, tmp.name, not args.no_display, args.save_debug,
                                     debug_log_fh) for v in videos]
        else:
            results = [process_video(args.source, model, is_head_model, tmp.name, not args.no_display, args.save_debug,
                                     debug_log_fh)]
    finally:
        Path(tmp.name).unlink(missing_ok=True)
        if debug_log_fh: debug_log_fh.close()

    save_log(results, args.log)


if __name__ == "__main__":
    main()