
import re
import base64
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st


def get_project_root() -> Path:
    root = Path(__file__).resolve().parent
    assert (root / "modelExperiments" / "DDR").exists(), (
        f"Project root validation failed: {root / 'modelExperiments' / 'DDR'} "
        "does not exist. Run the app from the ai/ folder."
    )
    return root


PROJECT_ROOT = get_project_root()
DDR_ROOT = PROJECT_ROOT / "modelExperiments" / "DDR"
MODEL_PATH = DDR_ROOT / "outputs_ddr_effnetb0" / "models" / "best_effnetb0_ddr.pth"

APP_OUTPUTS = DDR_ROOT / "streamlit_app_outputs"
UPLOADS_DIR = APP_OUTPUTS / "uploads"
OVERLAYS_DIR = APP_OUTPUTS / "overlays"
MASKS_DIR = APP_OUTPUTS / "masks"
DB_PATH = APP_OUTPUTS / "retina_predictions.db"

IMAGE_SIZE = 512

FIDELITY_TOP_PERCENTAGES = (0.05, 0.10, 0.15, 0.20)


def ensure_output_dirs() -> None:
    for d in (APP_OUTPUTS, UPLOADS_DIR, OVERLAYS_DIR, MASKS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def rel_to_outputs(path: Path) -> str:
    return Path(path).resolve().relative_to(APP_OUTPUTS.resolve()).as_posix()


def abs_from_outputs(rel_path: str) -> Path:
    return APP_OUTPUTS / rel_path


@st.cache_resource(show_spinner="Loading EfficientNet-B0 model…")
def load_predictor():
    from src.inference.predict import RetinaPredictor

    return RetinaPredictor()


def get_cam_object(predictor, method: str):
    from src.explainability.gradcam import GradCAM
    from src.explainability.gradcam_pp import GradCAMPlusPlus
    from src.explainability.x_gradcam import XGradCAM

    current = st.session_state.get("cam_method")
    if current != method or "cam_object" not in st.session_state:
        old = st.session_state.get("cam_object")
        if old is not None:
            try:
                old.remove_hooks()
            except Exception:
                pass

        target_layer = predictor.model.backbone.blocks[6]
        if method == "Grad-CAM++":
            cam_obj = GradCAMPlusPlus(predictor.model, target_layer, predictor.device)
        elif method == "X-Grad-CAM":
            cam_obj = XGradCAM(predictor.model, target_layer, predictor.device)
        else:
            cam_obj = GradCAM(predictor.model, target_layer, predictor.device)

        st.session_state["cam_object"] = cam_obj
        st.session_state["cam_method"] = method

    return st.session_state["cam_object"]


def predict_class(predictor, input_tensor):
    import torch

    with torch.no_grad():
        outputs = predictor.model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())

    prob_map = {predictor.class_names[i]: float(probs[i].item())
                for i in range(len(predictor.class_names))}
    return idx, conf, prob_map


def retina_mask(original_image: np.ndarray, tolerance: int = 7) -> np.ndarray:
    gray = cv2.cvtColor((original_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return gray > tolerance


def mask_cam_to_retina(cam: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = cam.copy()
    out[~mask] = 0.0
    return out


def make_heatmap(cam: np.ndarray) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def overlay_bytes(predictor, original_image: np.ndarray, cam: np.ndarray) -> bytes:
    b64 = predictor.create_heatmap_overlay(original_image, cam)
    return base64.b64decode(b64)


def compute_image_xai_metrics(predictor, cam_obj, original_image, cam, class_idx,
                              original_confidence):
    cam_norm = predictor.normalize_cam(cam)

    fidelity_by_p = {f"fidelity_top{int(p * 100)}": 0.0 for p in FIDELITY_TOP_PERCENTAGES}
    fidelity = 0.0
    try:
        drops = predictor.fidelity_percentile_drops(
            original_image, cam, class_idx, original_confidence,
            top_percentages=FIDELITY_TOP_PERCENTAGES,
        )
        fidelity_by_p = {f"fidelity_top{int(p * 100)}": round(float(d), 4)
                         for p, d in drops.items()}
        fidelity = round(float(drops.get(0.20, next(iter(drops.values())))), 4)
    except Exception:
        pass

    stability = 0.0
    stability_corr = 0.0
    try:
        from skimage.metrics import structural_similarity as ssim
        perturbed = predictor.perturb_image(original_image)
        perturbed_tensor = predictor.image_to_tensor(perturbed)
        p_cam, _, _ = cam_obj.generate(
            input_tensor=perturbed_tensor,
            class_idx=class_idx,
            image_size=IMAGE_SIZE,
        )
        p_cam = predictor.normalize_cam(p_cam)

        s = ssim(cam_norm, p_cam, data_range=1.0)
        stability = 0.0 if np.isnan(s) else float(np.clip(s, 0.0, 1.0))

        corr = np.corrcoef(cam_norm.flatten(), p_cam.flatten())[0, 1]
        stability_corr = 0.0 if np.isnan(corr) else float(np.clip(corr, -1.0, 1.0))
    except Exception:
        stability = 0.0
        stability_corr = 0.0

    try:
        deletion_auc, insertion_auc = predictor.get_auc_metrics(
            image=original_image, cam=cam_norm, target_class=class_idx,
            n_segments=100, baseline_mode="blur",
        )
    except Exception:
        deletion_auc, insertion_auc = 0.0, 0.0

    return {
        "fidelity": round(float(fidelity), 4),
        **fidelity_by_p,
        "stability": round(float(stability), 4),
        "stability_corr": round(float(stability_corr), 4),
        "deletion_auc": round(float(deletion_auc), 4),
        "insertion_auc": round(float(insertion_auc), 4),
    }


def cam_to_binary(cam: np.ndarray, keep_top_percent: float = 10.0) -> np.ndarray:
    thr = np.percentile(cam, 100.0 - keep_top_percent)
    return (cam >= thr).astype(np.uint8)


def dice_iou(cam_bin: np.ndarray, expert_mask: np.ndarray):
    cam_bool = cam_bin > 0
    mask_bool = expert_mask > 0

    intersection = np.logical_and(cam_bool, mask_bool).sum()
    union = np.logical_or(cam_bool, mask_bool).sum()

    cam_area = cam_bool.sum()
    mask_area = mask_bool.sum()

    dice = (2 * intersection) / (cam_area + mask_area + 1e-8)
    iou = intersection / (union + 1e-8)
    return float(dice), float(iou)


_SAM_COL_RE = re.compile(r"^\s*sam[_ ]?(point|box|pt|bx)", re.IGNORECASE)

_METRIC_RE = re.compile(r"^\s*([A-Za-z0-9_ /+().%-]+?)\s*:\s*([-+]?\d*\.?\d+)\s*$")

_SECTION_RE = re.compile(r"^\s*=+\s*(.+?)\s*=+\s*$")


def _pretty_section(name: str) -> str:
    key = re.sub(r"[\s_+-]", "", name).lower()
    if key == "gradcam":
        return "Grad-CAM"
    if key in ("gradcampp", "gradcamplusplus"):
        return "Grad-CAM++"
    return name.strip()

DASHBOARD_SECTIONS = [
    ("Classification report & accuracy", ["classification_report"]),
    ("F2 / F3 metrics", ["f2_f3"]),
    ("Confusion matrix", ["confusion_matrix"]),
    ("Fidelity & stability", ["fidelity_stability"]),
    ("Insertion / deletion AUC", ["slic_insertion_deletion", "insertion_deletion"]),
    ("Statistical significance tests", ["significance"]),
    ("Localization (MAPLES/MESSIDOR): Dice, IoU, Pointing Game, Energy-PG",
     ["dice_iou_pointing", "xai_comparison"]),
]


EXCLUDED_REPORT_FILES = {
    "confusion_matrix_test_f2_f3_DDR.png",
    "fidelity_stability_500_results_DDR_test.txt",
    "slic_insertion_deletion_auc_DDR_test.txt",
    "fidelity_per_image_DDR_test.csv",
    "stability_per_image_DDR_test.csv",
}


def scan_report_files():
    files = []
    if not DDR_ROOT.exists():
        return files
    for top in sorted(DDR_ROOT.iterdir()):
        if not top.is_dir():
            continue
        if not (top.name.startswith("outputs") or top.name.startswith("reports")):
            continue
        for f in sorted(top.rglob("*")):
            if f.name in EXCLUDED_REPORT_FILES:
                continue
            if f.is_file() and f.suffix.lower() in {".txt", ".csv", ".png"}:
                files.append(f)
    return files


def fmt3(value) -> str:
    s = str(value).strip()
    try:
        f = float(s)
    except (ValueError, TypeError):
        return s
    if "." not in s and "e" not in s.lower():
        return s
    return f"{f:.3f}"


_FAMILY_RULES = [
    ("Fidelity", lambda l: "fidelity" in l),
    ("Stability", lambda l: "stability" in l or "corr" in l or "ssim" in l),
    ("Deletion AUC", lambda l: "deletion" in l),
    ("Insertion AUC", lambda l: "insertion" in l),
]


def metric_family(label: str) -> str:
    l = label.lower()
    for name, test in _FAMILY_RULES:
        if test(l):
            return name
    return "Other"


def group_metrics_by_family(metrics):
    from collections import OrderedDict
    groups = OrderedDict()
    for label, value in metrics:
        groups.setdefault(metric_family(label), []).append((label, value))
    return groups


def categorize_files(files):
    buckets = {title: [] for title, _ in DASHBOARD_SECTIONS}
    buckets["Other reports"] = []

    for f in files:
        name = f.name.lower()
        placed = False
        for title, keys in DASHBOARD_SECTIONS:
            if any(k in name for k in keys):
                buckets[title].append(f)
                placed = True
                break
        if not placed:
            buckets["Other reports"].append(f)
    return buckets


def drop_sam_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if not _SAM_COL_RE.match(str(c))]
    return df[keep]


def parse_txt_metrics(text: str):
    metrics = []
    section = None
    try:
        for line in text.splitlines():
            sm = _SECTION_RE.match(line)
            if sm:
                section = _pretty_section(sm.group(1))
                continue
            m = _METRIC_RE.match(line)
            if m:
                label = m.group(1).strip()
                value = m.group(2).strip()
                if section:
                    label = f"{section} · {label}"
                metrics.append((label, value))
    except Exception:
        return []
    return metrics