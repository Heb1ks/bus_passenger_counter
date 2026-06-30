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

COUNTING_LINE_Y_RATIO = 0.28
ZONE_MARGIN_RATIO = 0.06
EDGE_MARGIN_X_RATIO = 0.12

EXIT_LINE_Y_RATIO = 0.10
FALLBACK_IN_LINE_Y_RATIO = 0.45

MIN_TRACK_DIST_RATIO = 0.02
GHOST_RADIUS_RATIO = 0.09
MERGE_RADIUS_RATIO = 0.10

MIN_BOX_AREA_RATIO = 0.001
MAX_BOX_AREA_RATIO = 0.25
MIN_ASPECT = 0.4
MAX_ASPECT = 2.5

CONF = 0.25
IOU = 0.35
IMGSZ = 640
MIN_TRACK_FRAMES = 3
MAX_BAG_RATIO = 0.3
IOA_THRESH = 0.50


STARTUP_FRAMES = 45

TRACKER_CFG = {
    "tracker_type": "botsort",
    "track_high_thresh": 0.30,
    "track_low_thresh": 0.05,
    "new_track_thresh": 0.30,
    "track_buffer": 60,
    "match_thresh": 0.85,
    "fuse_score": True,
    "gmc_method": "sparseOptFlow",
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.25,
    "with_reid": False,
    "model": None,
}

COOLDOWN_FRAMES = 50
GHOST_MAX_GAP = 15
GHOST_MIN_ALIVE = 2

DEBUG = False
DEBUG_LOG_FILE = "debug_oneline.log"

VIDEOS_DIR = "2026_05_23/middle"
DEFAULT_VID = "videos/kakoetovideo.mp4"


def dbg(msg: str, log_fh=None):
    if DEBUG:
        print(f"[DBG] {msg}")
        if log_fh:
            log_fh.write(f"[DBG] {msg}\n")
            log_fh.flush()


def load_head_model(local_path: str = HEAD_MODEL_LOCAL) -> tuple[YOLO, bool]:
    p = Path(local_path)
    if p.exists():
        print(f"[MODEL] Успешно загружена локальная модель: {p}")
        model = YOLO(str(p))
        model.to("cuda")
        return model, True
    else:
        print(f"\n[ОШИБКА] Файл модели '{local_path}' не найден!")
        exit()


class GhostTracker:
    def __init__(self, radius: float, max_gap: int = GHOST_MAX_GAP, min_alive: int = GHOST_MIN_ALIVE):
        self.radius = radius
        self.max_gap = max_gap
        self.min_alive = min_alive
        self._ghosts: dict[int, dict] = {}
        self._restored: dict[int, int] = {}

    def update_lost(self, tid: int, cx: float, cy: float, frame_idx: int, alive_frames: int):
        if alive_frames >= self.min_alive:
            self._ghosts[tid] = {"cx": cx, "cy": cy, "frame": frame_idx, "alive": alive_frames}

    def try_restore(self, new_tid: int, cx: float, cy: float, frame_idx: int,
                    counted_in_ids: set, counted_out_ids: set) -> int | None:
        if new_tid in self._restored:
            return self._restored[new_tid]
        best_dist = self.radius + 1
        best_old = None
        for old_tid, g in self._ghosts.items():
            if old_tid in counted_in_ids or old_tid in counted_out_ids:
                continue
            gap = frame_idx - g["frame"]
            if gap > self.max_gap: continue
            d = np.hypot(cx - g["cx"], cy - g["cy"])
            if d < best_dist:
                best_dist = d
                best_old = old_tid
        if best_old is not None:
            self._restored[new_tid] = best_old
            del self._ghosts[best_old]
            return best_old
        return None

    def cleanup(self, frame_idx: int):
        expired = [k for k, v in self._ghosts.items() if frame_idx - v["frame"] > self.max_gap]
        for k in expired: self._ghosts.pop(k, None)


def draw_overlay(frame, fw, fh, ci, co, frame_idx, line_y, zone_m, edge_x, exit_line_y, fallback_in_line_y):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (edge_x, fh), (0, 0, 255), -1)
    cv2.rectangle(overlay, (fw - edge_x, 0), (fw, fh), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    cv2.line(frame, (0, exit_line_y), (fw, exit_line_y), (0, 0, 255), 2)
    cv2.putText(frame, "EXIT LINE final", (10, exit_line_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    cv2.line(frame, (0, fallback_in_line_y), (fw, fallback_in_line_y), (0, 165, 255), 2)
    cv2.putText(frame, "ENTRY LINE final ", (10, fallback_in_line_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

    cv2.line(frame, (0, line_y), (fw, line_y), (0, 255, 0), 3)
    cv2.line(frame, (0, line_y - zone_m), (fw, line_y - zone_m), (0, 200, 0), 2)
    cv2.line(frame, (0, line_y + zone_m), (fw, line_y + zone_m), (0, 200, 0), 2)

    cv2.putText(frame, "DEAD ZONE", (10, line_y - zone_m - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.putText(frame, f"IN : {ci}", (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 230, 0), 2)
    cv2.putText(frame, f"OUT: {co}", (16, 84), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 100, 230), 2)
    cv2.putText(frame, f"f={frame_idx}", (fw - 120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def process_video(source: str, model: YOLO, is_head_model: bool,
                  tracker_path: str, show: bool, save_debug: bool,
                  debug_log_fh=None) -> dict:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): return {}

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0

    TARGET_WIDTH = 1024
    if fw > TARGET_WIDTH:
        scale = TARGET_WIDTH / fw
        fw = TARGET_WIDTH
        fh = int(fh * scale)

    line_y = int(fh * COUNTING_LINE_Y_RATIO)
    exit_line_y = int(fh * EXIT_LINE_Y_RATIO)
    fallback_in_line_y = int(fh * FALLBACK_IN_LINE_Y_RATIO)
    zone_m = int(fh * ZONE_MARGIN_RATIO)
    edge_x = int(fw * EDGE_MARGIN_X_RATIO)
    track_dist = int(fh * MIN_TRACK_DIST_RATIO)
    ghost_radius = int(fw * GHOST_RADIUS_RATIO)
    merge_radius = int(fw * MERGE_RADIUS_RATIO)

    frame_area = fw * fh
    min_box_area = frame_area * MIN_BOX_AREA_RATIO
    max_box_area = frame_area * MAX_BOX_AREA_RATIO

    dummy = np.zeros((fh, fw, 3), dtype=np.uint8)
    for _ in range(3): model.predict(dummy, verbose=False, imgsz=IMGSZ)

    detect_classes = [0]
    writer = None
    if save_debug:
        out_path = str(Path(source).stem) + "_debug_oneline.mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

    counted_in_ids: set[int] = set()
    counted_out_ids: set[int] = set()

    last_counted_frame: dict[int, int] = defaultdict(lambda: -(COOLDOWN_FRAMES + 1))
    track_alive: dict[int, int] = defaultdict(int)
    last_known_pos: dict[int, tuple[float, float]] = {}

    first_seen_frame: dict[int, int] = {}
    first_seen_pos: dict[int, tuple[float, float]] = {}
    track_zone: dict[int, str] = {}

    ghost = GhostTracker(radius=ghost_radius)
    tid_map: dict[int, int] = {}

    count_in = count_out = 0
    events = []
    frame_idx = 0
    show_window = show
    prev_real_tids: set[int] = set()

    # Умное окно для огромных видео
    if show_window:
        cv2.namedWindow("Smart One-Line Counter", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Smart One-Line Counter", 1280, 720)

    def log_event(action: str, tid: int):
        nonlocal count_in, count_out
        ts = datetime.now().strftime("%H:%M:%S")
        events.append((ts, action, tid))
        msg = f"  [{ts}] {action}  id={tid}  IN={count_in} OUT={count_out}"
        print(msg)
        if debug_log_fh:
            debug_log_fh.write(msg + "\n")
            debug_log_fh.flush()

    track_min_cy: dict[int, float] = {}

    crossed_entry_line: set[int] = set()
    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame.shape[1] > TARGET_WIDTH:
            frame = cv2.resize(frame, (fw, fh))

        results = model.track(
            source=frame, classes=detect_classes,
            conf=CONF, iou=IOU, imgsz=IMGSZ,
            tracker=tracker_path, persist=True, verbose=False, augment=True, quantize = 16,
            device="cuda"
        )

        real_tids: set[int] = set()
        sized = []

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            for xyxy, tid_raw in zip(results[0].boxes.xyxy.cpu().numpy(),
                                     results[0].boxes.id.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = xyxy
                w_box, h_box = x2 - x1, y2 - y1
                area, aspect = w_box * h_box, w_box / max(h_box, 1)
                cx = (x1 + x2) / 2.0
                cy = y2 - (y2 - y1) * 0.38  # Опускаем точку трекинга вниз (к шее/плечам)

                if cx < edge_x or cx > (fw - edge_x): continue
                if area < min_box_area or area > max_box_area or not (MIN_ASPECT <= aspect <= MAX_ASPECT): continue

                if tid_raw not in tid_map:
                    old_tid = ghost.try_restore(tid_raw, cx, cy, frame_idx,
                                                counted_in_ids, counted_out_ids)
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
                is_bag_nearby = (dist < merge_radius) and (min_area / max_area < MAX_BAG_RATIO)

                if is_overlap or is_bag_nearby:
                    suppressed.add(sized[j][1] if area_i >= area_j else sized[i][1])
                    if area_i < area_j: break

        for xyxy, tid, cx, cy, area in sized:
            if tid in suppressed: continue
            x1, y1, x2, y2 = xyxy
            track_alive[tid] += 1
            last_known_pos[tid] = (cx, cy)

            if tid not in first_seen_frame:
                first_seen_frame[tid] = frame_idx
                first_seen_pos[tid] = (cx, cy)

            cooldown_ok = (frame_idx - last_counted_frame[tid]) > COOLDOWN_FRAMES

            frames_alive = frame_idx - first_seen_frame[tid]
            start_x, start_y = first_seen_pos[tid]
            total_disp = np.hypot(cx - start_x, cy - start_y)

            is_valid_track = (frames_alive >= MIN_TRACK_FRAMES) and (total_disp >= track_dist)

            if cy < (line_y - zone_m):
                current_zone = "STREET"
            elif cy > (line_y + zone_m):
                current_zone = "BUS"
            else:
                current_zone = track_zone.get(tid, "UNKNOWN")

            prev_zone = track_zone.get(tid, "UNKNOWN")
            track_zone[tid] = current_zone

            # НОВОЕ: Обновляем минимальную позицию трека за всё время
            if tid not in track_min_cy:
                track_min_cy[tid] = cy
            else:
                track_min_cy[tid] = min(track_min_cy[tid], cy)


            never_was_above_exit = track_min_cy.get(tid, cy) > exit_line_y

            dy = cy - start_y
            total_dy_from_start = cy - start_y
            final_moving_up = total_dy_from_start < -fh * 0.05

            spawned_at_street = start_y < line_y
            spawned_at_bus = start_y > (line_y + zone_m)

            crossed_in = (
                                 (prev_zone == "STREET" and current_zone == "BUS") or
                                 (prev_zone == "STREET" and current_zone == "UNKNOWN" and dy > track_dist) or
                                 (
                                             prev_zone == "UNKNOWN" and current_zone == "BUS" and spawned_at_street and dy > track_dist)
                         ) and not final_moving_up

            crossed_out = (prev_zone == "BUS" and current_zone == "STREET") or \
                          (prev_zone == "BUS" and current_zone == "UNKNOWN" and dy < -track_dist)

            appeared_at_startup = first_seen_frame[tid] < STARTUP_FRAMES

            moving_into_bus = dy > track_dist * 0.5     # уверенно вниз
            moving_into_street = dy < -track_dist * 0.5  # уверенно вверх


            spawned_in_deadzone = (line_y - zone_m) <= start_y <= (line_y + zone_m)

            can_count_in = (
                    spawned_at_street
                    or (appeared_at_startup and moving_into_bus and never_was_above_exit)
                    or (spawned_in_deadzone and moving_into_bus)
                    or (spawned_at_bus and moving_into_bus and never_was_above_exit and frames_alive < 30)
            )
            can_count_out = (
                    spawned_at_bus
                    or (appeared_at_startup and moving_into_street)
                    or (appeared_at_startup and crossed_out)
                    or (spawned_in_deadzone and moving_into_street)
            )

            # if tid == 1:
            #     print(f"  [DEBUG-IN] tid={tid} f={frame_idx} "
            #           f"prev={prev_zone} cur={current_zone} "
            #           f"crossed_in={crossed_in} can_in={can_count_in} "
            #           f"spawned_street={spawned_at_street} "
            #           f"dy={dy:.1f} track_dist={track_dist} "
            #           f"final_up={final_moving_up} "
            #           f"valid={is_valid_track} cooldown={cooldown_ok}")

            if crossed_in and cooldown_ok and is_valid_track and can_count_in and (tid not in counted_in_ids):
                counted_in_ids.add(tid)
                last_counted_frame[tid] = frame_idx
                count_in += 1
                log_event("IN", tid)
                cv2.circle(frame, (int(cx), int(cy)), 15, (0, 255, 0), -1)

            elif crossed_out and cooldown_ok and is_valid_track and can_count_out and (tid not in counted_out_ids):
                counted_out_ids.add(tid)
                last_counted_frame[tid] = frame_idx
                count_out += 1
                log_event("OUT", tid)
                cv2.circle(frame, (int(cx), int(cy)), 15, (0, 80, 255), -1)

            # if tid == 1:
            #     print(f"  [DEBUG-OUT] tid={tid} f={frame_idx} "
            #           f"cy={cy:.1f} exit_line={exit_line_y} "
            #           f"total_dy={total_dy_from_start:.1f} threshold={-fh * 0.08:.1f} "
            #           f"counted_out={tid in counted_out_ids} "
            #           f"counted_in={tid in counted_in_ids} "
            #           f"crossed_out={crossed_out} can_out={can_count_out} "
            #           f"spawned_bus={spawned_at_bus} startup={appeared_at_startup}")

            # if tid == 37:
            #     print(f"  [DEBUG-OUT] tid={tid} f={frame_idx} "
            #           f"cy={cy:.1f} exit_line={exit_line_y} "
            #           f"total_dy={total_dy_from_start:.1f} threshold={-fh * 0.08:.1f} "
            #           f"counted_out={tid in counted_out_ids} "
            #           f"counted_in={tid in counted_in_ids} "
            #           f"crossed_out={crossed_out} can_out={can_count_out} "
            #           f"spawned_bus={spawned_at_bus} startup={appeared_at_startup}")
            spawned_in_street_or_dz = start_y < (line_y + zone_m)

            if ((spawned_in_street_or_dz or (spawned_at_bus and never_was_above_exit
                                             and frames_alive < 30
                                             and total_dy_from_start > fh * 0.15
                                             and not appeared_at_startup))
                    and tid not in counted_in_ids
                    and tid not in counted_out_ids
                    and tid not in crossed_entry_line
                    and cy > fallback_in_line_y
                    and dy > track_dist * 0.5
                    and is_valid_track
                    and cooldown_ok):
                crossed_entry_line.add(tid)
                counted_in_ids.add(tid)
                last_counted_frame[tid] = frame_idx
                count_in += 1
                log_event("IN(entryline)", tid)
                cv2.circle(frame, (int(cx), int(cy)), 15, (0, 165, 255), -1)

            if (cy < exit_line_y
                    and tid not in counted_out_ids
                    and tid not in counted_in_ids
                    and total_dy_from_start < -fh * 0.10):
                counted_out_ids.add(tid)
                last_counted_frame[tid] = frame_idx
                count_out += 1
                log_event("OUT(exitline)", tid)
                cv2.circle(frame, (int(cx), int(cy)), 15, (0, 0, 255), -1)

            is_counted = (tid in counted_in_ids) or (tid in counted_out_ids)
            box_color = (0, 200, 200) if is_counted else (200, 200, 0)

            if not is_valid_track:
                box_color = (0, 0, 255)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"#{tid}", (int(x1), max(int(y1) - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

        prev_real_tids = {s[1] for s in sized if s[1] not in suppressed}

        draw_overlay(frame, fw, fh, count_in, count_out, frame_idx, line_y, zone_m, edge_x, exit_line_y, fallback_in_line_y)

        if writer: writer.write(frame)
        frame_idx += 1

        if show_window:
            try:
                cv2.imshow("Smart One-Line Counter", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
            except cv2.error:
                show_window = False

    cap.release()
    if writer: writer.release()
    if show_window: cv2.destroyAllWindows()

    return {"file": source, "in": count_in, "out": count_out, "events": events}


def save_log(results, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Отчёт счётчика пассажиров (Universal Resolution Adaptive)\n")
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
    global DEBUG, STARTUP_FRAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_VID)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--save-debug", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--model", default=HEAD_MODEL_LOCAL)
    ap.add_argument("--log", default="oneline_log.txt")
    ap.add_argument("--startup-frames", type=int, default=STARTUP_FRAMES,
                    help="Кадры стартового окна (по умолчанию 45)")
    args = ap.parse_args()



    if args.debug: DEBUG = True
    STARTUP_FRAMES = args.startup_frames

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