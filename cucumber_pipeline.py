"""
cucumber_pipeline.py
====================
One-file pipeline: sorted frame folder → annotated output video.

Sections
--------
A  Imports and configuration
B  Helper utilities
C  SAM interface (with HSV-based fallback when SAM/PyTorch unavailable)
D  Contour extraction and filtering
E  Measurement functions
F  Tracker
G  Annotation and output
H  Main loop
"""

# =============================================================================
# SECTION A — Imports and configuration
# =============================================================================

import os
import sys
import math
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
from scipy.ndimage import label as scipy_label
from skimage.morphology import skeletonize

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("cucumber_pipeline")

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_FOLDER   = os.path.join(os.path.dirname(__file__), "Vegetables")
OUTPUT_VIDEO   = os.path.join(os.path.dirname(__file__), "cucumber_output.avi")
SAM_CHECKPOINT = ""          # path to SAM .pth file; leave empty to use fallback
SAM_MODEL_TYPE = "vit_b"     # "vit_b" | "vit_l" | "vit_h"

# ── Video settings ─────────────────────────────────────────────────────────────
OUTPUT_FPS     = 20
DISPLAY_LIVE   = True        # set False for headless / CI environments
WAITKEY_MS     = 50          # ms to pause between displayed frames

# ── Orientation ───────────────────────────────────────────────────────────────
# Supported values: 0 (none), 90, 180, 270  (degrees clockwise)
ROTATION_ANGLE = 0

# ── SAM settings (used only if SAM is available) ──────────────────────────────
SAM_POINTS_PER_SIDE   = 32
SAM_PRED_IOU_THRESH   = 0.86
SAM_STAB_SCORE_THRESH = 0.92
SAM_MIN_MASK_REGION   = 1000

# ── Contour filtering thresholds ─────────────────────────────────────────────
MIN_CONTOUR_AREA   = 6_000     # pixels²  – reject tiny blobs
MAX_CONTOUR_AREA   = 200_000   # pixels²  – reject background-size regions
MIN_ASPECT_RATIO   = 2.5       # long/short – cucumbers are elongated; relaxed for curved ones
MAX_CIRCULARITY    = 0.52      # 4π·A/P²  – reject round blobs
MIN_SOLIDITY       = 0.50      # A / hull_area; relaxed so partial cucumbers pass
MAX_SOLIDITY       = 1.00
HSV_FILTER_ENABLED = True
# Primary HSV range for cucumber-green colour (H in OpenCV 0-179)
HSV_PRIMARY_LOW  = np.array([22,  30,  40])
HSV_PRIMARY_HIGH = np.array([95, 255, 255])
# Retain short aliases so the rest of the code compiles unchanged
HSV_LOW  = HSV_PRIMARY_LOW
HSV_HIGH = HSV_PRIMARY_HIGH
# Secondary HSV range – darker/shadowed cucumber regions
HSV_LOW2  = np.array([18,  20,  20])
HSV_HIGH2 = np.array([100, 255, 140])
HSV_MIN_GREEN_FRACTION = 0.20  # relaxed: shadowed cucumbers have less saturated green

# ── Tracker thresholds ────────────────────────────────────────────────────────
TRACK_MAX_DIST        = 160    # px  – max centroid jump per frame
TRACK_MAX_AREA_RATIO  = 6.0    # loose area check; noisy masks vary a lot
TRACK_SHAPE_SIM_MAX   = 999.0  # effectively disabled – noisy masks are not shape-comparable
TRACK_MIN_HIT_FRAMES  = 6      # count a cucumber after 6 stable frames (reduces fragment overcounting)
TRACK_MAX_MISS_FRAMES = 10     # keep identity across short segmentation gaps
TRACK_MIN_DISPLAY_HITS = 2     # show a track after 2 hits

# ── Spatial dedup at count time (used only for stable-track mode) ─────────────
# If a second track gets "counted" within this many pixels of an already-counted
# centroid, we suppress it (it's a re-detection of the same physical cucumber).
# NOTE: for conveyor scenes COUNTING_LINE_X should be None (stable-track mode)
# since cucumbers move through all x-positions and line-crossing spatial dedup
# would incorrectly block new cucumbers at the same position as old ones.
TRACK_COUNT_MIN_DIST  = 80     # px; must be < distance between adjacent cucumbers

# ── Display / rolling window ──────────────────────────────────────────────────
DISPLAY_MEDIAN_WINDOW = 20    # number of recent frames used for rolling median

# ── Counting line ─────────────────────────────────────────────────────────────
# Set to None to auto-derive from frame width (mid-frame vertical line).
# For a conveyor moving left→right, mid-frame is the natural crossing point.
COUNTING_LINE_X: Optional[int] = None   # None = auto (set to frame_width // 2 at runtime)

# ── Visualisation toggles ────────────────────────────────────────────────────
DRAW_CONTOUR     = True
DRAW_FILL        = True         # semi-transparent mask fill
DRAW_CENTROID    = True
DRAW_AXIS        = True
DRAW_LENGTH      = True
DRAW_THICKNESS   = True
DRAW_CURVATURE   = True
DRAW_TRACK_TRAIL = False        # disabled – reduces visual clutter
DRAW_COUNT       = True

# ── Label appearance ──────────────────────────────────────────────────────────
LABEL_FONT_SCALE  = 0.52        # text size
LABEL_FONT_THICK  = 1           # text stroke thickness
LABEL_LINE_HEIGHT = 18          # pixels between label lines
LABEL_PAD         = 4           # padding inside background box
LABEL_BG_ALPHA    = 0.65        # opacity of the dark background box (0–1)


# =============================================================================
# SECTION B — Helper utilities
# =============================================================================

def load_frame_paths(folder: str) -> List[str]:
    """Return sorted list of image paths from *folder*."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    paths = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1].lower() in exts
    ]
    if not paths:
        raise FileNotFoundError(f"No images found in: {folder}")
    log.info("Found %d frames in %s", len(paths), folder)
    return paths


def rectify_frame(frame: np.ndarray, angle: int = ROTATION_ANGLE) -> np.ndarray:
    """Rotate *frame* by *angle* degrees clockwise (0/90/180/270)."""
    if angle == 0:
        return frame
    codes = {90: cv2.ROTATE_90_CLOCKWISE,
             180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    code = codes.get(angle % 360)
    if code is None:
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
        return cv2.warpAffine(frame, M, (w, h))
    return cv2.rotate(frame, code)


def create_video_writer(path: str, width: int, height: int,
                        fps: float = OUTPUT_FPS) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    return writer


def point_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def sort_endpoints(pts: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Given an array of 2-D points return the two that are furthest apart."""
    pts = pts.reshape(-1, 2)
    best_d, best_i, best_j = 0.0, 0, 1
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = point_distance(pts[i], pts[j])
            if d > best_d:
                best_d, best_i, best_j = d, i, j
    return tuple(pts[best_i]), tuple(pts[best_j])


def poly_arc_length(pts: np.ndarray) -> float:
    """Arc length of a polyline defined by ordered points (N×2)."""
    if len(pts) < 2:
        return 0.0
    d = np.diff(pts, axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


# =============================================================================
# SECTION C — SAM interface
# =============================================================================

_SAM_PREDICTOR = None
_SAM_AVAILABLE = False


def _try_init_sam() -> bool:
    """Attempt to initialise SAM; return True on success."""
    global _SAM_PREDICTOR, _SAM_AVAILABLE
    if _SAM_AVAILABLE:
        return True
    if not SAM_CHECKPOINT or not os.path.isfile(SAM_CHECKPOINT):
        return False
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        _SAM_PREDICTOR = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=SAM_POINTS_PER_SIDE,
            pred_iou_thresh=SAM_PRED_IOU_THRESH,
            stability_score_thresh=SAM_STAB_SCORE_THRESH,
            min_mask_region_area=SAM_MIN_MASK_REGION,
        )
        _SAM_AVAILABLE = True
        log.info("SAM initialised successfully.")
        return True
    except Exception as exc:
        log.warning("SAM init failed (%s); using fallback segmentation.", exc)
        return False


@dataclass
class CandidateMask:
    binary: np.ndarray    # uint8 single-channel 0/255
    area: float
    bbox: Tuple[int, int, int, int]   # x, y, w, h
    score: float = 1.0


def _sam_segment(frame: np.ndarray) -> List[CandidateMask]:
    """Run SAM on *frame* and return candidate masks."""
    masks_data = _SAM_PREDICTOR.generate(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results: List[CandidateMask] = []
    for m in masks_data:
        seg = (m["segmentation"].astype(np.uint8)) * 255
        area = float(m["area"])
        bbox = tuple(int(v) for v in m["bbox"])   # x, y, w, h
        score = float(m.get("predicted_iou", 1.0))
        results.append(CandidateMask(binary=seg, area=area, bbox=bbox, score=score))
    return results


def _fallback_segment(frame: np.ndarray) -> List[CandidateMask]:
    """
    Improved fallback segmentation using:
    - CLAHE equalisation to handle reflections and shadows
    - Two HSV ranges (bright green + shadowed green)
    - Per-component mask smoothing via Gaussian blur + re-threshold
    Mimics SAM by returning a list of per-object CandidateMask objects.
    """
    # ── 1. CLAHE on L channel to normalise lighting ──────────────────────
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    equalized = cv2.merge([l_ch, a_ch, b_ch])
    enhanced  = cv2.cvtColor(equalized, cv2.COLOR_LAB2BGR)

    # ── 2. HSV on enhanced frame (primary range) ─────────────────────────
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    mask_a = cv2.inRange(hsv, HSV_LOW,  HSV_HIGH)

    # ── 3. HSV on original frame (catches areas CLAHE over-saturates) ────
    hsv_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_b   = cv2.inRange(hsv_orig, HSV_LOW,  HSV_HIGH)
    mask_c   = cv2.inRange(hsv_orig, HSV_LOW2, HSV_HIGH2)

    combined = cv2.bitwise_or(mask_a, cv2.bitwise_or(mask_b, mask_c))

    # ── 4. Morphological cleanup ─────────────────────────────────────────
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed  = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close, iterations=3)
    opened  = cv2.morphologyEx(closed,   cv2.MORPH_OPEN,  k_open,  iterations=2)

    # ── 5. Per-component mask smoothing ──────────────────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        opened, connectivity=8)
    results: List[CandidateMask] = []
    for lbl in range(1, num_labels):
        area_raw = float(stats[lbl, cv2.CC_STAT_AREA])
        if area_raw < MIN_CONTOUR_AREA * 0.4:
            continue
        blob = (labels == lbl).astype(np.uint8) * 255
        # Smooth each blob independently (fills small holes, removes spurs)
        blob = cv2.GaussianBlur(blob, (7, 7), 0)
        _, blob = cv2.threshold(blob, 64, 255, cv2.THRESH_BINARY)
        # Final area after smoothing
        area = float((blob > 0).sum())
        if area < MIN_CONTOUR_AREA * 0.4:
            continue
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        results.append(CandidateMask(binary=blob, area=area, bbox=(x, y, w, h)))
    return results


def get_candidate_masks(frame: np.ndarray) -> List[CandidateMask]:
    """
    Return candidate object masks for *frame*.
    Uses SAM when available; otherwise falls back to colour-based segmentation.
    """
    if _SAM_AVAILABLE:
        return _sam_segment(frame)
    return _fallback_segment(frame)


# =============================================================================
# SECTION D — Contour extraction and filtering
# =============================================================================

@dataclass
class Detection:
    """All descriptors for one accepted cucumber candidate in a single frame."""
    frame_idx:      int
    contour:        np.ndarray
    hull:           np.ndarray
    binary_mask:    np.ndarray
    bbox:           Tuple[int, int, int, int]    # x, y, w, h (upright)
    rotated_box:    cv2.RotatedRect              # (centre, (w,h), angle)
    area:           float
    perimeter:      float
    circularity:    float
    solidity:       float
    centroid:       Tuple[float, float]
    axis_angle_deg: float                        # orientation in degrees
    axis_vec:       Tuple[float, float]          # unit direction vector
    length:         float
    thickness:      float
    curvature:      float
    length_pts:     Tuple[Tuple[int, int], Tuple[int, int]]
    thickness_pts:  Tuple[Tuple[int, int], Tuple[int, int]]
    hsv_stats:      Optional[np.ndarray] = None  # mean HSV inside mask


def _smooth_contour(contour: np.ndarray, epsilon_frac: float = 0.008) -> np.ndarray:
    """
    Simplify/smooth a contour with approxPolyDP then re-interpolate via convex-hull
    blending to remove micro-jitter while preserving overall shape.
    """
    peri = cv2.arcLength(contour, True)
    eps  = epsilon_frac * peri
    approx = cv2.approxPolyDP(contour, eps, True)
    # Ensure minimum 5 points so ellipse fit can work
    if len(approx) < 5:
        return contour
    return approx


def _mask_to_best_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Extract the largest valid contour from a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Keep the largest contour by area
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < MIN_CONTOUR_AREA * 0.3:
        return None
    return _smooth_contour(best)


def _dedup_masks(masks: List[CandidateMask]) -> List[CandidateMask]:
    """Remove highly overlapping duplicate masks; keep highest-score one."""
    if len(masks) <= 1:
        return masks
    keep = []
    used = [False] * len(masks)
    # Sort by score descending so the better mask wins
    order = sorted(range(len(masks)), key=lambda i: -masks[i].score)
    for i in order:
        if used[i]:
            continue
        keep.append(masks[i])
        for j in order:
            if i == j or used[j]:
                continue
            # IoU check
            inter = np.logical_and(masks[i].binary, masks[j].binary).sum()
            union = np.logical_or(masks[i].binary, masks[j].binary).sum()
            iou = inter / (union + 1e-9)
            if iou > 0.35:          # lower than 0.5 default to catch near-duplicate masks
                used[j] = True
    return keep


def _passes_hsv_filter(frame: np.ndarray, mask: np.ndarray) -> bool:
    """Return True if enough of the masked region is cucumber-green (either range)."""
    if not HSV_FILTER_ENABLED:
        return True
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_a = cv2.inRange(hsv, HSV_LOW,  HSV_HIGH)
    green_b = cv2.inRange(hsv, HSV_LOW2, HSV_HIGH2)
    green   = cv2.bitwise_or(green_a, green_b)
    valid   = mask > 0
    total_px = int(valid.sum())
    if total_px == 0:
        return False
    green_px = int((green[valid] > 0).sum())
    return (green_px / total_px) >= HSV_MIN_GREEN_FRACTION


def extract_and_filter_contours(
        frame: np.ndarray,
        masks: List[CandidateMask],
        frame_idx: int) -> List[Detection]:
    """
    Convert SAM masks → contours, compute descriptors, filter cucumber-like shapes.
    Returns accepted Detection objects.
    """
    masks = _dedup_masks(masks)
    detections: List[Detection] = []

    for cm in masks:
        # ── Extract contour ────────────────────────────────────────────────
        contour = _mask_to_best_contour(cm.binary)
        if contour is None:
            continue

        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue

        peri = cv2.arcLength(contour, True)
        if peri < 1:
            continue

        # ── Convex hull ────────────────────────────────────────────────────
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-9)

        # ── Circularity ────────────────────────────────────────────────────
        circularity = (4 * math.pi * area) / (peri * peri + 1e-9)

        # ── Aspect ratio from rotated rect ─────────────────────────────────
        if len(contour) < 5:
            continue
        rect = cv2.minAreaRect(contour)   # (centre, (w,h), angle)
        rw, rh = rect[1]
        if min(rw, rh) < 1:
            continue
        aspect = max(rw, rh) / (min(rw, rh) + 1e-9)

        # ── Geometric filters ──────────────────────────────────────────────
        if aspect < MIN_ASPECT_RATIO:
            continue
        if circularity > MAX_CIRCULARITY:
            continue
        if not (MIN_SOLIDITY <= solidity <= MAX_SOLIDITY):
            continue

        # ── HSV colour filter ──────────────────────────────────────────────
        mask_u8 = (cm.binary > 0).astype(np.uint8) * 255
        if not _passes_hsv_filter(frame, mask_u8):
            continue

        # ── If we reach here, contour is accepted ─────────────────────────
        det = _build_detection(frame, frame_idx, contour, hull, cm.binary, rect,
                               area, peri, circularity, solidity)
        if det is not None:
            detections.append(det)

    # ── Second-pass dedup: drop any detection whose centroid is too close
    #    to a larger already-accepted detection (catches masks the IoU pass misses).
    detections = _dedup_detections_by_centroid(detections)
    return detections


def _dedup_detections_by_centroid(
        detections: List["Detection"],
        min_dist: int = 60) -> List["Detection"]:
    """Keep at most one detection per spatial cluster (nearest-centroid)."""
    if len(detections) <= 1:
        return detections
    # Sort largest area first so the best representative wins
    dets = sorted(detections, key=lambda d: -d.area)
    kept: List["Detection"] = []
    for det in dets:
        too_close = any(
            point_distance(det.centroid, k.centroid) < min_dist
            for k in kept
        )
        if not too_close:
            kept.append(det)
    return kept


def _build_detection(
        frame: np.ndarray, frame_idx: int,
        contour: np.ndarray, hull: np.ndarray,
        binary_mask: np.ndarray,
        rect,
        area: float, peri: float,
        circularity: float, solidity: float) -> Optional[Detection]:
    """Compute all geometry and assemble a Detection object."""
    # Bounding box
    bx, by, bw, bh = cv2.boundingRect(contour)

    # Centroid from moments
    M = cv2.moments(contour)
    if M["m00"] < 1:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # Main axis from ellipse fit (primary) / rotated rect (fallback)
    axis_angle_deg, axis_vec = _estimate_axis(contour, rect)

    # Measurements
    length, thickness, curvature, len_pts, thick_pts = _measure(
        contour, binary_mask, (cx, cy), axis_vec)

    # HSV stats inside mask
    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_roi = hsv_frame[mask_u8 > 0]
    hsv_stats = hsv_roi.mean(axis=0) if len(hsv_roi) > 0 else None

    return Detection(
        frame_idx=frame_idx,
        contour=contour,
        hull=hull,
        binary_mask=binary_mask.astype(np.uint8),
        bbox=(bx, by, bw, bh),
        rotated_box=rect,
        area=area,
        perimeter=peri,
        circularity=circularity,
        solidity=solidity,
        centroid=(cx, cy),
        axis_angle_deg=axis_angle_deg,
        axis_vec=axis_vec,
        length=length,
        thickness=thickness,
        curvature=curvature,
        length_pts=len_pts,
        thickness_pts=thick_pts,
        hsv_stats=hsv_stats,
    )


# =============================================================================
# SECTION E — Measurement functions
# =============================================================================

def _estimate_axis(contour: np.ndarray,
                   rect) -> Tuple[float, Tuple[float, float]]:
    """
    Estimate the cucumber's dominant orientation.
    Primary: ellipse fit. Fallback: rotated rectangle long side.
    Returns (angle_degrees, unit_vector).
    """
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            angle = ellipse[2]               # OpenCV: angle of MINOR axis from x-axis
            # Convert to major-axis direction
            major_angle_rad = math.radians(angle + 90)
            vx = math.cos(major_angle_rad)
            vy = math.sin(major_angle_rad)
            return angle + 90, (vx, vy)
        except cv2.error:
            pass

    # Fallback: rotated rect
    (cx, cy), (rw, rh), angle = rect
    if rw < rh:
        angle += 90
    angle_rad = math.radians(angle)
    return angle, (math.cos(angle_rad), math.sin(angle_rad))


def _extract_centerline(binary_mask: np.ndarray,
                        min_length: int = 10) -> Optional[np.ndarray]:
    """
    Skeletonise the binary mask and return ordered centerline points (N×2)
    in the original mask coordinate space.

    Improvements over the old greedy walk:
    - Gaussian-smooth the mask before skeletonising (fills holes, reduces spurs)
    - Build explicit 8-connectivity graph, identify true endpoint pixels
    - Trace the longest path between endpoints (no side-branch jumping)
    - Scale back to original coordinates
    Returns None if centerline is too short or unreliable.
    """
    h, w = binary_mask.shape[:2]
    # Downsample so the largest dimension is at most 200 px
    scale = min(1.0, 200.0 / max(h, w, 1))
    sh, sw = max(1, int(h * scale)), max(1, int(w * scale))
    small = cv2.resize(binary_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)

    # Smooth before skeletonising to fill interior holes and remove edge spurs
    small = cv2.GaussianBlur(small, (5, 5), 0)
    _, small = cv2.threshold(small, 64, 255, cv2.THRESH_BINARY)

    bw   = (small > 0).astype(np.uint8)
    skel = skeletonize(bw).astype(np.uint8)

    ys, xs = np.where(skel > 0)
    if len(xs) < min_length:
        return None

    pts = np.column_stack([xs, ys])

    # ── Graph-based endpoint tracing ─────────────────────────────────────
    ordered = _trace_skeleton_path(pts)
    if ordered is None or len(ordered) < min_length:
        return None

    # Scale back to original coordinates
    if scale < 1.0:
        ordered = (ordered.astype(float) / scale).astype(np.int32)
    return ordered


def _trace_skeleton_path(pts: np.ndarray) -> Optional[np.ndarray]:
    """
    Build an 8-connectivity adjacency structure from skeleton pixels.
    Find the two pixels with fewest neighbours (true endpoints).
    Do a BFS/DFS to find the path between them that covers the most pixels,
    avoiding side branches.  Returns an ordered (N×2) array.
    """
    if len(pts) < 2:
        return None

    # Index pixels into a set for O(1) lookup
    pts_set: dict = {(int(p[0]), int(p[1])): i for i, p in enumerate(pts)}
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def neighbours(p):
        x, y = p
        return [(x+dx, y+dy) for dx, dy in dirs if (x+dx, y+dy) in pts_set]

    # Degree of each pixel
    degree = {tuple(p): len(neighbours(tuple(p))) for p in pts}

    # Endpoints: degree 1 (or 0 for isolated, treat as both endpoint)
    endpoints = [p for p, d in degree.items() if d <= 1]
    if len(endpoints) < 2:
        # No clear endpoints (e.g. closed loop) – pick the two extremes along x
        xs_arr = pts[:, 0]
        endpoints = [tuple(pts[xs_arr.argmin()]), tuple(pts[xs_arr.argmax()])]

    # Walk from the endpoint with smallest x to the one with largest x
    # (ensures consistent left→right direction)
    endpoints.sort(key=lambda p: p[0])
    start = endpoints[0]
    goal  = endpoints[-1]

    # Greedy walk following the path (at each step prefer unvisited with lowest degree)
    ordered = [start]
    visited = {start}
    current = start
    while True:
        nbrs = [n for n in neighbours(current) if n not in visited]
        if not nbrs:
            break
        # Prefer the neighbour that is closest to the goal (greedy A*-lite)
        gx, gy = goal
        nbrs.sort(key=lambda n: (n[0]-gx)**2 + (n[1]-gy)**2)
        nxt = nbrs[0]
        ordered.append(nxt)
        visited.add(nxt)
        current = nxt
        if current == goal:
            break

    return np.array(ordered) if len(ordered) >= 2 else None


def _order_skeleton_points(pts: np.ndarray) -> Optional[np.ndarray]:
    """Legacy wrapper – delegates to _trace_skeleton_path."""
    return _trace_skeleton_path(pts)


def _measure(
        contour: np.ndarray,
        binary_mask: np.ndarray,
        centroid: Tuple[float, float],
        axis_vec: Tuple[float, float]
) -> Tuple[float, float, float,
           Tuple[Tuple[int, int], Tuple[int, int]],
           Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Compute length, thickness, curvature.
    Primary: centerline-based with local tangent for thickness.
    Fallback: principal-axis projection.
    Returns (length, thickness, curvature, length_pts, thickness_pts).

    Curvature = arc_length / chord_length (≥ 1.0 by definition).
    Clamped to [1.0, 2.0] — values outside that range indicate a
    broken centerline and are discarded in favour of 1.0.
    """
    # ── Try centerline approach ─────────────────────────────────────────────
    cl = _extract_centerline(binary_mask)
    if cl is not None and len(cl) >= 8:
        arc   = poly_arc_length(cl.astype(float))
        ep1   = tuple(cl[0].astype(int))
        ep2   = tuple(cl[-1].astype(int))
        chord = point_distance(ep1, ep2)

        raw_curv = arc / (chord + 1e-9)
        # Sanity-clamp: a broken skeleton produces arc >> chord
        curvature = float(np.clip(raw_curv, 1.0, 2.0)) if chord > 5 else 1.0

        # Use local tangent at midpoint for thickness direction
        mid_idx = len(cl) // 2
        mid_pt  = cl[mid_idx].astype(float)
        # Estimate local tangent from a window of ±3 points
        lo = max(0, mid_idx - 3)
        hi = min(len(cl) - 1, mid_idx + 3)
        tangent = cl[hi].astype(float) - cl[lo].astype(float)
        t_len = math.hypot(tangent[0], tangent[1])
        if t_len > 1e-3:
            local_vec = (tangent[0] / t_len, tangent[1] / t_len)
        else:
            local_vec = axis_vec

        thickness, tp1, tp2 = _perpendicular_thickness(
            contour, mid_pt, local_vec)

        return arc, thickness, curvature, (ep1, ep2), (tp1, tp2)

    # ── Fallback: principal axis from PCA on contour points ────────────────
    pts    = contour.reshape(-1, 2).astype(np.float32)
    mean   = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
    principal  = vt[0]
    projections = (pts - mean) @ principal
    min_p, max_p = projections.min(), projections.max()
    ep1 = tuple((mean + min_p * principal).astype(int))
    ep2 = tuple((mean + max_p * principal).astype(int))
    arc = point_distance(ep1, ep2)
    curvature = 1.0   # no bending info from axis projection

    thickness, tp1, tp2 = _perpendicular_thickness(
        contour, np.array(centroid), axis_vec)

    return arc, thickness, curvature, (ep1, ep2), (tp1, tp2)


def _perpendicular_thickness(
        contour: np.ndarray,
        mid_pt: np.ndarray,
        axis_vec: Tuple[float, float],
        search_len: int = 300
) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
    """
    Cast a ray perpendicular to *axis_vec* from *mid_pt* and intersect
    the contour on both sides to measure thickness.
    """
    vx, vy = axis_vec
    # Perpendicular direction
    px, py = -vy, vx

    # Sample points along perpendicular ray
    best_d = 0.0
    tp1 = tuple(mid_pt.astype(int))
    tp2 = tuple(mid_pt.astype(int))

    # Build a filled contour mask to intersect the perpendicular ray with.
    # Dimensions must cover both the contour extents and the mid_pt position.
    cy_max = max(int(contour[:, :, 1].max()), int(mid_pt[1])) + search_len + 10
    cx_max = max(int(contour[:, :, 0].max()), int(mid_pt[0])) + search_len + 10
    mask = np.zeros((cy_max, cx_max), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    h, w = mask.shape

    def _intersect(direction: int) -> Optional[Tuple[int, int]]:
        for t in range(1, search_len):
            qx = int(round(mid_pt[0] + direction * t * px))
            qy = int(round(mid_pt[1] + direction * t * py))
            if qx < 0 or qy < 0 or qx >= w or qy >= h:
                return None
            if mask[qy, qx] == 0:
                # Just stepped outside the contour
                qx = int(round(mid_pt[0] + direction * (t - 1) * px))
                qy = int(round(mid_pt[1] + direction * (t - 1) * py))
                return (qx, qy)
        return None

    side_a = _intersect(+1)
    side_b = _intersect(-1)

    if side_a and side_b:
        best_d = point_distance(side_a, side_b)
        tp1, tp2 = side_a, side_b
    elif side_a:
        best_d = point_distance(tuple(mid_pt.astype(int)), side_a) * 2
        tp1 = side_a
    elif side_b:
        best_d = point_distance(tuple(mid_pt.astype(int)), side_b) * 2
        tp2 = side_b

    return best_d, tp1, tp2


# =============================================================================
# SECTION F — Tracker
# =============================================================================

@dataclass
class Track:
    """Persistent state for one tracked cucumber."""
    track_id:       int
    centroid:       Tuple[float, float]
    prev_centroid:  Optional[Tuple[float, float]] = None
    centroid_hist:  List[Tuple[float, float]] = field(default_factory=list)
    contour_hist:   List[np.ndarray]           = field(default_factory=list)
    hull_hist:      List[np.ndarray]           = field(default_factory=list)
    area_hist:      List[float]                = field(default_factory=list)
    length_hist:    List[float]                = field(default_factory=list)
    thickness_hist: List[float]               = field(default_factory=list)
    curvature_hist: List[float]               = field(default_factory=list)
    first_frame:    int  = 0
    last_frame:     int  = 0
    miss_count:     int  = 0
    hit_count:      int  = 0
    counted:        bool = False
    # crossing line state
    last_cx:        Optional[float] = None
    # current display values (medians)
    display_length:    float = 0.0
    display_thickness: float = 0.0
    display_curvature: float = 1.0
    current_detection: Optional[Detection] = None


class Tracker:
    """Simple multi-object centroid tracker with ID counting."""

    def __init__(self) -> None:
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._total_count = 0
        self._counted_centroids: List[Tuple[float, float]] = []

    @property
    def total_count(self) -> int:
        return self._total_count

    @property
    def active_tracks(self) -> List[Track]:
        return list(self._tracks.values())

    def update(self, detections: List[Detection], frame_idx: int) -> List[Track]:
        """
        Associate *detections* to existing tracks.
        Returns list of currently active (updated) tracks.
        """
        if not detections:
            self._expire_tracks(frame_idx)
            return self.active_tracks

        track_ids  = list(self._tracks.keys())
        track_list = [self._tracks[tid] for tid in track_ids]

        if not track_list:
            # No existing tracks → create one per detection
            for det in detections:
                self._new_track(det, frame_idx)
            return self.active_tracks

        # ── Build cost matrix (lower = better) ────────────────────────────
        n_det, n_trk = len(detections), len(track_list)
        cost = np.full((n_det, n_trk), np.inf)
        for di, det in enumerate(detections):
            for ti, trk in enumerate(track_list):
                dist = point_distance(det.centroid, trk.centroid)
                if dist > TRACK_MAX_DIST:
                    continue
                # Area similarity
                a1, a2 = det.area, (trk.area_hist[-1] if trk.area_hist else det.area)
                area_ratio = max(a1, a2) / (min(a1, a2) + 1e-9)
                if area_ratio > TRACK_MAX_AREA_RATIO:
                    continue
                # Shape similarity (Hu moments on log scale)
                shape_d = 0.0
                if trk.contour_hist:
                    shape_d = cv2.matchShapes(
                        det.contour, trk.contour_hist[-1],
                        cv2.CONTOURS_MATCH_I2, 0.0)
                    if shape_d > TRACK_SHAPE_SIM_MAX:
                        continue
                cost[di, ti] = dist + 10 * shape_d + 0.1 * area_ratio

        # ── Greedy assignment ──────────────────────────────────────────────
        matched_det  = set()
        matched_trk  = set()
        while True:
            if np.all(np.isinf(cost)):
                break
            idx = np.unravel_index(np.argmin(cost), cost.shape)
            di, ti = idx
            if np.isinf(cost[di, ti]):
                break
            matched_det.add(di)
            matched_trk.add(ti)
            cost[di, :] = np.inf
            cost[:, ti] = np.inf
            trk = track_list[ti]
            det = detections[di]
            self._update_track(trk, det, frame_idx)

        # ── Unmatched detections → new tracks ─────────────────────────────
        for di, det in enumerate(detections):
            if di not in matched_det:
                self._new_track(det, frame_idx)

        # ── Unmatched tracks → increment miss counter ──────────────────────
        for ti, trk in enumerate(track_list):
            if ti not in matched_trk:
                trk.miss_count += 1

        self._expire_tracks(frame_idx)
        self._try_count(frame_idx)
        return self.active_tracks

    # ── Internal helpers ───────────────────────────────────────────────────

    def _new_track(self, det: Detection, frame_idx: int) -> Track:
        trk = Track(
            track_id=self._next_id,
            centroid=det.centroid,
            first_frame=frame_idx,
            last_frame=frame_idx,
        )
        self._next_id += 1
        self._update_track(trk, det, frame_idx)
        self._tracks[trk.track_id] = trk
        return trk

    def _update_track(self, trk: Track, det: Detection, frame_idx: int) -> None:
        trk.prev_centroid = trk.centroid
        trk.centroid = det.centroid
        trk.centroid_hist.append(det.centroid)
        trk.contour_hist.append(det.contour)
        trk.hull_hist.append(det.hull)
        trk.area_hist.append(det.area)
        trk.length_hist.append(det.length)
        trk.thickness_hist.append(det.thickness)
        trk.curvature_hist.append(det.curvature)
        trk.last_frame = frame_idx
        trk.miss_count = 0
        trk.hit_count += 1
        trk.current_detection = det
        # Update rolling median display values
        if trk.length_hist:
            trk.display_length    = float(np.median(trk.length_hist[-DISPLAY_MEDIAN_WINDOW:]))
            trk.display_thickness = float(np.median(trk.thickness_hist[-DISPLAY_MEDIAN_WINDOW:]))
            trk.display_curvature = float(np.median(trk.curvature_hist[-DISPLAY_MEDIAN_WINDOW:]))
        trk.last_cx = det.centroid[0]

    def _expire_tracks(self, frame_idx: int) -> None:
        expired = [tid for tid, t in self._tracks.items()
                   if t.miss_count >= TRACK_MAX_MISS_FRAMES]
        for tid in expired:
            del self._tracks[tid]

    def _try_count(self, frame_idx: int) -> None:
        """
        Count each track once when it first reaches TRACK_MIN_HIT_FRAMES.
        This is the most robust strategy for conveyor scenes:
        - No line-crossing needed (cucumbers may already be past any fixed line)
        - No spatial dedup (cucumbers move through the same x-positions over time)
        - trk.counted=True prevents ever counting the same track twice
        The only double-count risk is fragmentation: one cucumber → 2+ tracks both
        reaching MIN_HIT_FRAMES. Mitigated by improved tracker stability (SHAPE
        check disabled, generous miss tolerance) rather than post-hoc dedup.
        """
        for trk in self._tracks.values():
            if trk.counted:
                continue
            if trk.hit_count >= TRACK_MIN_HIT_FRAMES:
                trk.counted = True
                self._total_count += 1


# =============================================================================
# SECTION G — Annotation and output
# =============================================================================

COLOURS = [
    (0, 230, 0),    # bright green
    (0, 190, 255),  # sky blue
    (255, 110, 0),  # orange
    (200, 0, 255),  # purple
    (0, 255, 200),  # cyan-green
    (255, 220, 0),  # yellow
    (180, 255, 0),  # lime
    (255, 0, 140),  # hot pink
]


def _track_colour(track_id: int) -> Tuple[int, int, int]:
    return COLOURS[track_id % len(COLOURS)]


def _draw_label_box(img: np.ndarray,
                    lines: List[str],
                    anchor: Tuple[int, int],
                    colour: Tuple[int, int, int],
                    frame_w: int, frame_h: int) -> None:
    """
    Draw a dark-background label block at *anchor* (top-left of box),
    clamped to stay inside the frame.
    """
    font       = cv2.FONT_HERSHEY_SIMPLEX
    scale      = LABEL_FONT_SCALE
    thick      = LABEL_FONT_THICK
    pad        = LABEL_PAD
    line_h     = LABEL_LINE_HEIGHT

    # Measure total box size
    max_w = max(cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines)
    box_w = max_w + 2 * pad
    box_h = line_h * len(lines) + 2 * pad

    # Clamp anchor so box stays on screen
    ax = max(0, min(anchor[0], frame_w - box_w - 1))
    ay = max(0, min(anchor[1], frame_h - box_h - 1))

    # Dark background with alpha blend
    overlay = img.copy()
    cv2.rectangle(overlay, (ax, ay), (ax + box_w, ay + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, LABEL_BG_ALPHA, img, 1 - LABEL_BG_ALPHA, 0, img)

    # Text lines
    for k, txt in enumerate(lines):
        ty = ay + pad + (k + 1) * line_h - 3
        cv2.putText(img, txt, (ax + pad, ty), font, scale, colour, thick + 1,
                    cv2.LINE_AA)
        cv2.putText(img, txt, (ax + pad, ty), font, scale, (255, 255, 255),
                    thick, cv2.LINE_AA)


def annotate_frame(frame: np.ndarray,
                   tracks: List[Track],
                   total_count: int) -> np.ndarray:
    """Draw all overlays onto *frame* and return the annotated copy."""
    out = frame.copy()
    frame_h, frame_w = out.shape[:2]

    # Only draw tracks that have been seen enough times (confirmed tracks)
    visible = [t for t in tracks
               if t.current_detection is not None
               and t.hit_count >= TRACK_MIN_DISPLAY_HITS]

    # ── Per-track drawing ──────────────────────────────────────────────────
    # First pass: fills (semi-transparent) so outlines render on top
    if DRAW_FILL:
        overlay = out.copy()
        for trk in visible:
            det = trk.current_detection
            colour = _track_colour(trk.track_id)
            cv2.drawContours(overlay, [det.contour], -1, colour, -1)
        cv2.addWeighted(overlay, 0.20, out, 0.80, 0, out)

    # Second pass: outlines, lines, centroids
    for trk in visible:
        det    = trk.current_detection
        colour = _track_colour(trk.track_id)
        cx, cy = int(det.centroid[0]), int(det.centroid[1])

        # Contour & hull
        if DRAW_CONTOUR:
            cv2.drawContours(out, [det.contour], -1, colour, 2)
            cv2.drawContours(out, [det.hull], -1,
                             tuple(max(0, c - 60) for c in colour), 1)

        # Centroid
        if DRAW_CENTROID:
            cv2.circle(out, (cx, cy), 6, (255, 255, 255), -1)
            cv2.circle(out, (cx, cy), 6, colour, 2)

        # Track trail (optional, off by default)
        if DRAW_TRACK_TRAIL and len(trk.centroid_hist) >= 2:
            trail = trk.centroid_hist[-15:]
            for k in range(1, len(trail)):
                p1 = (int(trail[k-1][0]), int(trail[k-1][1]))
                p2 = (int(trail[k][0]),   int(trail[k][1]))
                cv2.line(out, p1, p2, colour, 1, cv2.LINE_AA)

        # Length line (cyan)
        if DRAW_LENGTH and det.length_pts:
            lp0, lp1 = det.length_pts
            cv2.line(out, lp0, lp1, (0, 230, 230), 2, cv2.LINE_AA)
            cv2.circle(out, lp0, 5, (0, 230, 230), -1)
            cv2.circle(out, lp1, 5, (0, 230, 230), -1)

        # Thickness line (orange)
        if DRAW_THICKNESS and det.thickness_pts:
            tp0, tp1 = det.thickness_pts
            cv2.line(out, tp0, tp1, (0, 130, 255), 2, cv2.LINE_AA)
            cv2.circle(out, tp0, 5, (0, 130, 255), -1)
            cv2.circle(out, tp1, 5, (0, 130, 255), -1)

    # Counting line
    if COUNTING_LINE_X is not None:
        cv2.line(out, (COUNTING_LINE_X, 0), (COUNTING_LINE_X, frame_h),
                 (0, 0, 230), 2)

    # ── Third pass: labels (drawn last so they sit on top of everything) ──
    # Collect preferred anchor positions and resolve collisions
    label_regions: List[Tuple[int, int, int, int]] = []   # x, y, w, h

    for trk in visible:
        det    = trk.current_detection
        colour = _track_colour(trk.track_id)
        cx, cy = int(det.centroid[0]), int(det.centroid[1])

        lines = [
            f"ID:{trk.track_id}",
            f"L:{trk.display_length:.0f}px",
            f"W:{trk.display_thickness:.0f}px",
            f"C:{trk.display_curvature:.2f}",
        ]

        font  = cv2.FONT_HERSHEY_SIMPLEX
        max_w = max(cv2.getTextSize(l, font, LABEL_FONT_SCALE,
                                    LABEL_FONT_THICK)[0][0] for l in lines)
        box_w = max_w + 2 * LABEL_PAD
        box_h = LABEL_LINE_HEIGHT * len(lines) + 2 * LABEL_PAD

        # Preferred position: just to the right of the centroid, vertically centred
        ax = cx + 10
        ay = cy - box_h // 2

        # Clamp to frame
        ax = max(0, min(ax, frame_w - box_w - 1))
        ay = max(0, min(ay, frame_h - box_h - 1))

        # Push down to avoid overlap with already-placed boxes
        for (rx, ry, rw, rh) in label_regions:
            if ax < rx + rw and ax + box_w > rx:        # horizontal overlap
                if ay < ry + rh and ay + box_h > ry:    # vertical overlap
                    ay = ry + rh + 2                     # shift below

        # Re-clamp after push
        ay = max(0, min(ay, frame_h - box_h - 1))

        label_regions.append((ax, ay, box_w, box_h))
        _draw_label_box(out, lines, (ax, ay), colour, frame_w, frame_h)

    # ── Total cucumber count (top-left corner) ────────────────────────────
    if DRAW_COUNT:
        count_txt = f"Cucumbers counted: {total_count}"
        (tw, th), _ = cv2.getTextSize(count_txt, cv2.FONT_HERSHEY_SIMPLEX,
                                       1.0, 2)
        # Dark background for the count banner
        cv2.rectangle(out, (0, 0), (tw + 16, th + 20), (0, 0, 0), -1)
        cv2.putText(out, count_txt, (8, th + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2, cv2.LINE_AA)

    return out


# =============================================================================
# SECTION H — Main loop
# =============================================================================

def main() -> None:
    global COUNTING_LINE_X
    log.info("=== Cucumber detection pipeline starting ===")

    # ── Initialise SAM (non-fatal if unavailable) ──────────────────────────
    _try_init_sam()
    if not _SAM_AVAILABLE:
        log.info("Using HSV fallback segmentation (SAM not available).")

    # ── Load frames ────────────────────────────────────────────────────────
    frame_paths = load_frame_paths(INPUT_FOLDER)

    # ── Read first frame to configure the video writer ────────────────────
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise IOError(f"Cannot read first frame: {frame_paths[0]}")
    first_frame = rectify_frame(first_frame)
    h, w = first_frame.shape[:2]

    # ── Auto-set counting line to frame mid-width when explicitly requested ──
    # If COUNTING_LINE_X is None, use stable-track counting (default for most scenes).
    if COUNTING_LINE_X is not None:
        log.info("Counting line at x=%d.", COUNTING_LINE_X)
    else:
        log.info("Using stable-track counting strategy (COUNTING_LINE_X not set).")

    writer = create_video_writer(OUTPUT_VIDEO, w, h)
    log.info("Writing output to: %s  (%dx%d @ %d fps)", OUTPUT_VIDEO, w, h, OUTPUT_FPS)

    tracker = Tracker()
    results: Dict[int, Dict] = {}   # per-cucumber final results

    # ── Main frame loop ────────────────────────────────────────────────────
    for frame_idx, fpath in enumerate(frame_paths):
        # ── Load and rectify ───────────────────────────────────────────────
        frame = cv2.imread(fpath)
        if frame is None:
            log.warning("Skipping unreadable frame: %s", fpath)
            continue
        frame = rectify_frame(frame)

        # ── Segmentation (SAM or fallback) ─────────────────────────────────
        masks = get_candidate_masks(frame)

        # ── Contour extraction and filtering ──────────────────────────────
        detections = extract_and_filter_contours(frame, masks, frame_idx)

        # ── Tracker update ─────────────────────────────────────────────────
        active_tracks = tracker.update(detections, frame_idx)

        # ── Accumulate final results for counted tracks ────────────────────
        for trk in active_tracks:
            if trk.counted and trk.track_id not in results:
                results[trk.track_id] = {
                    "id": trk.track_id,
                    "first_frame": trk.first_frame,
                    "last_frame": trk.last_frame,
                    "length_px": trk.display_length,
                    "thickness_px": trk.display_thickness,
                    "curvature": trk.display_curvature,
                }

        # ── Annotation ─────────────────────────────────────────────────────
        annotated = annotate_frame(frame, active_tracks, tracker.total_count)

        # ── Write to video ─────────────────────────────────────────────────
        writer.write(annotated)

        # ── Optional live display ──────────────────────────────────────────
        if DISPLAY_LIVE:
            cv2.imshow("Cucumber Pipeline", annotated)
            key = cv2.waitKey(WAITKEY_MS)
            if key == 27:   # Esc
                log.info("User interrupted at frame %d.", frame_idx)
                break

        if (frame_idx + 1) % 20 == 0:
            log.info("Frame %d/%d | detections=%d | tracked=%d | count=%d",
                     frame_idx + 1, len(frame_paths),
                     len(detections), len(active_tracks), tracker.total_count)

    # ── Finalise ────────────────────────────────────────────────────────────
    writer.release()
    if DISPLAY_LIVE:
        cv2.destroyAllWindows()

    log.info("Total unique cucumbers counted: %d", tracker.total_count)

    # ── Print per-cucumber summary ─────────────────────────────────────────
    if results:
        log.info("%-6s %-12s %-14s %-10s %-10s",
                 "ID", "First Frame", "Last Frame", "Length(px)", "Thickness(px)")
        for rid, r in sorted(results.items()):
            log.info("%-6d %-12d %-14d %-10.1f %-10.1f  curvature=%.2f",
                     r["id"], r["first_frame"], r["last_frame"],
                     r["length_px"], r["thickness_px"], r["curvature"])

    log.info("Output video: %s", OUTPUT_VIDEO)
    log.info("=== Pipeline finished ===")


if __name__ == "__main__":
    main()
