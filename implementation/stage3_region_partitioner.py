"""
stage3_region_partitioner.py — Stage 3: 7-Region Anatomical Slicing & 4-Extreme-Point Compression
Paper: Quantum Information Entanglement Index for Facial Expression Analysis (ECML PKDD 2026)

This module handles:
- Partitioning 68 landmarks into 7 anatomical regions (jaw, brows, nose, eyes, mouth)
- Extracting the 4 extreme points (leftmost, rightmost, topmost, bottommost) -> 8 coordinates
- L2-normalization for quantum feature mapping
- Mathematical validation of bounding envelopes and normalization invariants
- Visual artifact generation: stage3_region_diamonds.png (bounding diamond polygons)
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# 1. REGIONAL INDICES & MAPPING DEFINITIONS
# ---------------------------------------------------------------------------
REGION_LANDMARK_MAP = {
    "jaw": list(range(0, 17)),
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow": list(range(22, 27)),
    "nose": list(range(27, 36)),
    "right_eye": list(range(36, 42)),
    "left_eye": list(range(42, 48)),
    "mouth_outer": list(range(48, 60)),
    "mouth_inner": list(range(60, 68))
}

REGION_COLORS = {
    "jaw": (231, 76, 60),           # Red
    "right_eyebrow": (155, 89, 182), # Purple
    "left_eyebrow": (142, 68, 173),  # Dark Purple
    "nose": (241, 196, 15),          # Yellow
    "right_eye": (52, 152, 219),     # Blue
    "left_eye": (41, 128, 185),      # Dark Blue
    "mouth_outer": (46, 204, 113),   # Green
    "mouth_inner": (39, 174, 96),    # Dark Green
}


# ---------------------------------------------------------------------------
# 2. EXTREME COORDINATES EXTRACTION & NORMALIZATION
# ---------------------------------------------------------------------------
def extract_region_extreme_points(points: np.ndarray) -> np.ndarray:
    """
    Finds the 4 extreme landmarks in a region:
      - Leftmost:   argmin(x)
      - Rightmost:  argmax(x)
      - Topmost:    argmin(y)
      - Bottommost: argmax(y)
    Returns a 1D float32 array of 8 coordinates: [x_L, y_L, x_R, y_R, x_T, y_T, x_B, y_B]
    """
    if len(points) == 0:
        return np.zeros(8, dtype=np.float32)

    p_left = points[np.argmin(points[:, 0])]
    p_right = points[np.argmax(points[:, 0])]
    p_top = points[np.argmin(points[:, 1])]
    p_bottom = points[np.argmax(points[:, 1])]

    return np.concatenate([p_left, p_right, p_top, p_bottom]).astype(np.float32)


def normalize_l2_vector(coords: np.ndarray) -> np.ndarray:
    """
    Performs L2 normalization on a coordinate vector: v_norm = v / ||v||_2.
    """
    norm = np.linalg.norm(coords)
    if norm > 1e-8:
        return coords / norm
    res = np.zeros_like(coords)
    res[0] = 1.0
    return res


def partition_and_extract_all_regions(landmarks_68: np.ndarray) -> dict:
    """
    Extracts raw and L2-normalized 8-coordinate extreme vectors for all 8 sub-regions.
    """
    region_data = {}
    for r_name, indices in REGION_LANDMARK_MAP.items():
        sub_pts = landmarks_68[indices]
        raw_extremes = extract_region_extreme_points(sub_pts)
        norm_extremes = normalize_l2_vector(raw_extremes)
        region_data[r_name] = {
            "indices": indices,
            "points": sub_pts,
            "raw_extremes": raw_extremes,
            "norm_extremes": norm_extremes
        }
    return region_data


# ---------------------------------------------------------------------------
# 3. VERIFICATION TESTS
# ---------------------------------------------------------------------------
def verify_extreme_points_envelope(landmarks_68: np.ndarray, region_data: dict) -> tuple:
    """
    Validates that:
    1. Every region has exactly 8 coordinates.
    2. All landmarks in region R lie strictly inside the bounding envelope:
       x_L <= x_i <= x_R and y_T <= y_i <= y_B.
    3. The normalized vector satisfies ||v||_2 = 1.0.
    """
    for r_name, data in region_data.items():
        pts = data["points"]
        ext = data["raw_extremes"]
        x_L, y_L, x_R, y_R, x_T, y_T, x_B, y_B = ext

        # 1. Length check
        if len(ext) != 8:
            return False, f"Region {r_name} has length {len(ext)} != 8"

        # 2. Envelope check
        if np.min(pts[:, 0]) < x_L - 1e-5:
            return False, f"Region {r_name}: landmark x smaller than x_L"
        if np.max(pts[:, 0]) > x_R + 1e-5:
            return False, f"Region {r_name}: landmark x greater than x_R"
        if np.min(pts[:, 1]) < y_T - 1e-5:
            return False, f"Region {r_name}: landmark y smaller than y_T"
        if np.max(pts[:, 1]) > y_B + 1e-5:
            return False, f"Region {r_name}: landmark y greater than y_B"

        # 3. L2 norm check
        norm_val = np.linalg.norm(data["norm_extremes"])
        if not np.isclose(norm_val, 1.0, atol=1e-5):
            return False, f"Region {r_name}: L2 norm is {norm_val:.6f} != 1.0"

    return True, "All 8 regions satisfy bounding envelope and L2 normalization"


def save_region_diamonds_preview(landmarks_68: np.ndarray, region_data: dict, output_path: str, width: int = 400, height: int = 400):
    """
    Renders visual bounding diamond polygons connecting:
    (x_L, y_L) -> (x_T, y_T) -> (x_R, y_R) -> (x_B, y_B) -> (x_L, y_L).
    """
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (width, height), color=(245, 247, 250))
        draw = ImageDraw.Draw(img)

        # Draw light background landmark dots
        for x, y in landmarks_68:
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(189, 195, 199))

        # Draw regional extreme diamond polygons
        for r_name, data in region_data.items():
            ext = data["raw_extremes"]
            color = REGION_COLORS.get(r_name, (52, 73, 94))
            p_L = (ext[0], ext[1])
            p_R = (ext[2], ext[3])
            p_T = (ext[4], ext[5])
            p_B = (ext[6], ext[7])

            diamond = [p_L, p_T, p_R, p_B]
            draw.polygon(diamond, outline=color, width=2)
            for px, py in diamond:
                draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=color)

        img.save(output_path)
        return True
    except Exception as e:
        print(f"Warning: Could not save diamond preview: {e}")
        return False


# ---------------------------------------------------------------------------
# STAGE 3 VERIFIABLE CHECKPOINT
# ---------------------------------------------------------------------------
def run_stage3_checkpoint() -> bool:
    print("=" * 65)
    print("STAGE 3: 7-REGION ANATOMICAL SLICING & DIAMOND EXTREMES CHECKPOINT")
    print("=" * 65)

    # Import Stage 2 detector for input coordinates
    from stage2_landmark_detector import detect_landmarks
    landmarks = detect_landmarks(None)

    region_data = partition_and_extract_all_regions(landmarks)

    # 1. Number of regions check
    num_regions = len(region_data)
    test1_pass = (num_regions == 8)
    print(f"  [{'PASS' if test1_pass else 'FAIL'}] Test 1: Partitioned into 8 regional sets (7 regions + outer/inner mouth)")

    # 2. Envelope & Normalization check
    test2_pass, msg = verify_extreme_points_envelope(landmarks, region_data)
    print(f"  [{'PASS' if test2_pass else 'FAIL'}] Test 2: Bounding envelopes and L2 norms: {msg}")

    # 3. Visual Artifact check
    artifact_dir = os.path.join(os.path.dirname(__file__), "artifacts", "stages")
    os.makedirs(artifact_dir, exist_ok=True)
    preview_file = os.path.join(artifact_dir, "stage3_region_diamonds.png")
    img_saved = save_region_diamonds_preview(landmarks, region_data, preview_file)
    print(f"  [{'PASS' if img_saved else 'FAIL'}] Test 3: Visual artifact saved: {preview_file}")

    all_passed = test1_pass and test2_pass and img_saved
    print("-" * 65)
    if all_passed:
        print("[STAGE 3 CHECKPOINT PASSED: ALL 7 REGIONAL EXTREMES VERIFIED]")
    else:
        print("[STAGE 3 CHECKPOINT FAILED]")
    print("=" * 65)
    return all_passed


if __name__ == "__main__":
    success = run_stage3_checkpoint()
    sys.exit(0 if success else 1)
