import sys
import cv2
import timm
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
import torch.nn as nn

from segment_anything import sam_model_registry, SamPredictor

HERE = Path(__file__).resolve()

AI_ROOT = None
for parent in HERE.parents:
    if (parent / "src").is_dir() and (parent / "modelExperiments").is_dir():
        AI_ROOT = parent
        break

if AI_ROOT is None:
    AI_ROOT = HERE.parents[2]

sys.path.insert(0, str(AI_ROOT))

from src.explainability.gradcam import GradCAM as GradCAMSimple
from src.explainability.gradcam_pp import GradCAMPlusPlus
from src.explainability.x_gradcam import XGradCAM


def init_sam(checkpoint_path, model_type="vit_h", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return SamPredictor(sam)

def refine_cam_with_sam_both(image_rgb_uint8, cam, sam_predictor, ktp=10.0):
    sam_predictor.set_image(image_rgb_uint8)

    y_max, x_max = np.unravel_index(np.argmax(cam), cam.shape)
    input_point = np.array([[x_max, y_max]])
    input_label = np.array([1])

    masks_pt, _, _ = sam_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )
    mask_point = masks_pt[0].astype(np.uint8)

    thr = np.percentile(cam, 100.0 - ktp)
    cam_bin = (cam >= thr).astype(np.uint8)
    coords = np.column_stack(np.where(cam_bin > 0))
    
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max_box, x_max_box = coords.max(axis=0)
        
        input_box = np.array([x_min, y_min, x_max_box, y_max_box])
        
        masks_box, _, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False
        )
        mask_box = masks_box[0].astype(np.uint8)
    else:
        mask_box = np.zeros_like(cam_bin)

    return mask_point, mask_box


class EfficientNetWithHead(nn.Module):
    def __init__(self, backbone, in_features, num_classes=5, dropout=0.5):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        pooled = self.pool(features)
        return self.head(pooled)

def build_model():
    backbone = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=0,
        drop_rate=0.3,
        drop_path_rate=0.2
    )
    in_features = backbone.num_features
    model = EfficientNetWithHead(
        backbone=backbone,
        in_features=in_features,
        num_classes=5,
        dropout=0.5
    )
    return model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512

MAPLES_ROOT = AI_ROOT / "data" / "MAPLES_DR"
MESSIDOR_ROOT = AI_ROOT / "data" / "MESSIDOR"

MODEL_PATH = (
    AI_ROOT
    / "modelExperiments"
    / "DDR"
    / "outputs_ddr_effnetb0"
    / "models"
    / "best_effnetb0_ddr.pth"
)

SAM_CHECKPOINT_PATH = AI_ROOT / "modelExperiments" / "sam_vit_b_01ec64.pth"

OUTPUT_DIR = AI_ROOT / "modelExperiments" / "DDR" / "outputs_maples_eval"
REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_CSV = REPORTS_DIR / "xai_comparison_summary.csv"
DETAILED_CSV = REPORTS_DIR / "xai_comparison_detailed.csv"

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
DR_LESIONS = ["CottonWoolSpots", "Exudates", "Hemorrhages", "Microaneurysms", "Neovascularization"]

METHODS = {
    "gradcam": GradCAMSimple,
    "gradcampp": GradCAMPlusPlus,
    "xgradcam": XGradCAM,
}

KTP_VALUES = [2.0, 3.0, 5.0, 10.0]
TARGET_LAYER_INDEX = 6
PG_TOLERANCES = [0, 15, 25]
DR_POSITIVE_GRADES = {"R1", "R2", "R3", "R4A", "R4S"}


def build_messidor_index():
    image_index = {}
    for ext in ["*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"]:
        for img_path in MESSIDOR_ROOT.rglob(ext):
            image_index[img_path.stem] = img_path
    return image_index

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = build_model()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image_path):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    image_float = image_rgb.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_norm = (image_float - mean) / std

    tensor = torch.tensor(image_norm, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE), image_float

def find_mask_file(lesion_dir, image_stem):
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        p = lesion_dir / f"{image_stem}{ext}"
        if p.exists():
            return p
    matches = list(lesion_dir.glob(f"{image_stem}.*"))
    if len(matches) > 0:
        return matches[0]
    return None

def get_combined_dr_mask(split, image_stem):
    combined_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    found_masks = []
    split_dir = MAPLES_ROOT / split

    for lesion in DR_LESIONS:
        lesion_dir = split_dir / lesion
        if not lesion_dir.exists():
            continue
        mask_path = find_mask_file(lesion_dir, image_stem)
        if mask_path is None:
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask_bin = (mask > 0).astype(np.uint8)
        combined_mask = np.maximum(combined_mask, mask_bin)
        found_masks.append(str(mask_path))

    return combined_mask, found_masks


def cam_to_binary(cam, keep_top_percent=10.0):
    thr = np.percentile(cam, 100.0 - keep_top_percent)
    return (cam >= thr).astype(np.uint8)

def dice_iou(cam_bin, expert_mask):
    cam_bool = cam_bin > 0
    mask_bool = expert_mask > 0
    intersection = np.logical_and(cam_bool, mask_bool).sum()
    union = np.logical_or(cam_bool, mask_bool).sum()
    dice = (2 * intersection) / (cam_bool.sum() + mask_bool.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    return float(dice), float(iou)

def pointing_game(max_coords, expert_mask, tolerance=0):
    y, x = max_coords
    if tolerance <= 0:
        return int(expert_mask[y, x] > 0)
    h, w = expert_mask.shape
    y0, y1 = max(0, y - tolerance), min(h, y + tolerance + 1)
    x0, x1 = max(0, x - tolerance), min(w, x + tolerance + 1)
    return int(expert_mask[y0:y1, x0:x1].any())

def energy_pointing_game(cam, expert_mask):
    cam = np.clip(cam, 0, None)
    total = float(cam.sum()) + 1e-8
    inside = float(cam[expert_mask > 0].sum())
    return inside / total


def evaluate_method(method_tag, MethodClass, model, messidor_index, sam_predictor, splits=("train", "test")):
    target_layer = model.backbone.blocks[TARGET_LAYER_INDEX]
    cam_obj = MethodClass(model=model, target_layer=target_layer, device=DEVICE)

    per_image = []

    for split in splits:
        diagnosis_path = MAPLES_ROOT / split / "diagnosis.csv"
        if not diagnosis_path.exists():
            print(f"  [!] Missing diagnosis.csv for split: {split}")
            continue

        df_diag = pd.read_csv(diagnosis_path)

        for _, row in tqdm(df_diag.iterrows(), total=len(df_diag), desc=f"  {method_tag} | {split}"):
            image_stem = str(row["name"])
            dr_label = row.get("DR", None)

            if str(dr_label).strip().upper() not in DR_POSITIVE_GRADES:
                continue

            image_path = messidor_index.get(image_stem)
            if image_path is None:
                continue

            try:
                input_tensor, image_float = preprocess_image(image_path)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                    pred_class = int(torch.argmax(probs).item())

                if pred_class == 0:
                    continue

                cam, _, max_coords = cam_obj.generate(
                    input_tensor=input_tensor,
                    class_idx=pred_class,
                    image_size=IMG_SIZE,
                    threshold=None
                )

                expert_mask, _ = get_combined_dr_mask(split, image_stem)
                if expert_mask.sum() == 0:
                    continue

                record = {
                    "method": method_tag,
                    "split": split,
                    "image_name": image_stem,
                    "DR": dr_label,
                    "predicted_class": CLASS_NAMES[pred_class],
                    "energy_pg": energy_pointing_game(cam, expert_mask),
                    "lesion_coverage": float((expert_mask > 0).mean()),
                }

                for tol in PG_TOLERANCES:
                    record[f"pg_t{tol}"] = pointing_game(max_coords, expert_mask, tolerance=tol)

                image_uint8 = (image_float * 255).astype(np.uint8)

                sam_mask_point, sam_mask_box = refine_cam_with_sam_both(
                    image_uint8, cam, sam_predictor, ktp=10.0
                )

                d_sam_pt, i_sam_pt = dice_iou(sam_mask_point, expert_mask)
                record["dice_sam_point"] = d_sam_pt
                record["iou_sam_point"] = i_sam_pt

                d_sam_box, i_sam_box = dice_iou(sam_mask_box, expert_mask)
                record["dice_sam_box"] = d_sam_box
                record["iou_sam_box"] = i_sam_box

                for ktp in KTP_VALUES:
                    cam_bin = cam_to_binary(cam, keep_top_percent=ktp)
                    d, i = dice_iou(cam_bin, expert_mask)
                    record[f"dice_ktp{ktp:g}"] = d
                    record[f"iou_ktp{ktp:g}"] = i

                per_image.append(record)

            except Exception as e:
                print(f"  [error] {image_stem}: {e}")
                continue

    cam_obj.remove_hooks()
    return per_image


def main(splits=("train", "test")):
    print("AI_ROOT (auto):", AI_ROOT)
    print("Device:", DEVICE)
    print("Target layer: model.backbone.blocks[%d]" % TARGET_LAYER_INDEX)
    print("Methods:", list(METHODS.keys()))
    print("keep_top_percent values:", KTP_VALUES)
    print("Output dir:", OUTPUT_DIR)

    if not SAM_CHECKPOINT_PATH.exists():
        print(f"\n[!] EROARE: Fisierul SAM nu a fost gasit la calea: {SAM_CHECKPOINT_PATH}")
        print("Te rog descarca 'sam_vit_b_01ec64.pth' si plaseaza-l in folderul indicat.")
        return

    messidor_index = build_messidor_index()
    print("MESSIDOR images indexed:", len(messidor_index))

    model = load_model()

    print("\nInitializare SAM Predictor...")
    sam_predictor = init_sam(checkpoint_path=str(SAM_CHECKPOINT_PATH), model_type="vit_b", device=DEVICE)

    all_records = []
    for method_tag, MethodClass in METHODS.items():
        print(f"\n=== Evaluez metoda: {method_tag} ===")
        recs = evaluate_method(
            method_tag=method_tag,
            MethodClass=MethodClass,
            model=model,
            messidor_index=messidor_index,
            sam_predictor=sam_predictor,
            splits=splits
        )
        all_records.extend(recs)

    df = pd.DataFrame(all_records)
    df.to_csv(DETAILED_CSV, index=False)

    comparison_rows = []
    for method_tag in METHODS.keys():
        sub = df[df["method"] == method_tag]
        if len(sub) == 0:
            continue
        
        sam_pt_dice = sub["dice_sam_point"].mean()
        sam_bx_dice = sub["dice_sam_box"].mean()
        sam_pt_iou = sub["iou_sam_point"].mean()
        sam_bx_iou = sub["iou_sam_box"].mean()
        
        for ktp in KTP_VALUES:
            comparison_rows.append({
                "method": method_tag,
                "keep_top_percent": ktp,
                "n_valid": len(sub),
                "mean_dice": round(sub[f"dice_ktp{ktp:g}"].mean(), 4),
                "mean_iou": round(sub[f"iou_ktp{ktp:g}"].mean(), 4),
                
                "sam_point_dice": round(sam_pt_dice, 4),
                "sam_box_dice": round(sam_bx_dice, 4),
                "sam_point_iou": round(sam_pt_iou, 4),
                "sam_box_iou": round(sam_bx_iou, 4),

                "pg_exact": round(sub["pg_t0"].mean(), 4),
                "pg_t15": round(sub["pg_t15"].mean(), 4),
                "pg_t25": round(sub["pg_t25"].mean(), 4),
                "energy_pg": round(sub["energy_pg"].mean(), 4),
                "lesion_coverage": round(sub["lesion_coverage"].mean(), 4),
            })

    df_cmp = pd.DataFrame(comparison_rows)
    df_cmp.to_csv(COMPARISON_CSV, index=False)

    print("\n" + "=" * 110)
    print("  TABEL COMPARATIV  (SAM Point/Box, Pointing Game, Energy)")
    print("=" * 110)
    header = (f"{'method':<11}{'ktp%':>6}{'Dice':>8}{'IoU':>8}"
              f"{'SAM_Pt_D':>10}{'SAM_Bx_D':>10}"
              f"{'PG_ex':>7}{'PG15':>7}{'PG25':>7}{'E-PG':>7}{'cover':>7}")
    print(header)
    print("-" * 110)
    for r in comparison_rows:
        print(f"{r['method']:<11}{r['keep_top_percent']:>6.0f}"
              f"{r['mean_dice']:>8.4f}{r['mean_iou']:>8.4f}"
              f"{r['sam_point_dice']:>10.4f}{r['sam_box_dice']:>10.4f}"
              f"{r['pg_exact']:>7.3f}{r['pg_t15']:>7.3f}{r['pg_t25']:>7.3f}"
              f"{r['energy_pg']:>7.3f}{r['lesion_coverage']:>7.4f}")
    print("=" * 110)

    print("\nSaved comparison table:", COMPARISON_CSV)
    print("Saved detailed per-image:", DETAILED_CSV)

if __name__ == "__main__":
    main(splits=("train", "test"))