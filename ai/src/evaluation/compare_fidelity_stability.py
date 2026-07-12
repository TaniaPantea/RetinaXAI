import cv2
import timm
import torch
import random
import numpy as np
import pandas as pd
import sys

from pathlib import Path
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from skimage.metrics import structural_similarity as ssim


HERE = Path(__file__).resolve()

AI_ROOT = None
for parent in HERE.parents:
    if (parent / "src").is_dir() and (parent / "modelExperiments").is_dir():
        AI_ROOT = parent
        break
if AI_ROOT is None:
    AI_ROOT = HERE.parents[2]

SRC_DIR = AI_ROOT / "src"
sys.path.insert(0, str(AI_ROOT))
sys.path.insert(0, str(SRC_DIR))

from explainability.gradcam import GradCAM as GradCAMSimple
from explainability.gradcam_pp import GradCAMPlusPlus
from explainability.x_gradcam import XGradCAM


SEED = 42
IMG_SIZE = 512
MAX_SAMPLES = 15000
TOP_PERCENTAGES = [0.05, 0.10, 0.15, 0.20]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DDR_DIR = AI_ROOT / "modelExperiments" / "DDR"

DATASET_DIR = AI_ROOT / "data" / "DR_grading"
IMAGE_DIR = DATASET_DIR

MODEL_PATH = DDR_DIR / "outputs_ddr_effnetb0" / "models" / "best_effnetb0_ddr.pth"
TEST_CSV = DDR_DIR / "outputs_ddr_effnetb0" / "splits" / "test.csv"

EVAL_OUTPUT_DIR = DDR_DIR / "outputs_fidelity_stability_eval"
REPORTS_DIR = EVAL_OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TXT_PATH = REPORTS_DIR / "fidelity_stability_comparison_DDR_test.txt"
CSV_PATH = REPORTS_DIR / "fidelity_stability_comparison_DDR_test.csv"

METHODS = {
    "gradcam": GradCAMSimple,
    "gradcampp": GradCAMPlusPlus,
    "xgradcam": XGradCAM,
}

TARGET_LAYER_INDEX = 6


VALID_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]



def resolve_image_path(image_folder, image_name):
    image_name = str(image_name)
    candidate = Path(image_name)

    if candidate.is_absolute() and candidate.exists():
        return candidate

    direct_path = Path(image_folder) / image_name
    if direct_path.exists():
        return direct_path

    if candidate.suffix.lower() == "":
        for ext in VALID_EXTENSIONS:
            path = Path(image_folder) / f"{image_name}{ext}"
            if path.exists():
                return path

    stem = candidate.stem
    for ext in VALID_EXTENSIONS:
        matches = list(Path(image_folder).rglob(f"{stem}{ext}"))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Image not found: {image_name} inside {image_folder}")


def crop_black_borders(image, tolerance=7):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > tolerance
    if mask.sum() == 0:
        return image
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    return image[y_min:y_max, x_min:x_max]


class FundusDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None, use_crop=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_folder = Path(image_folder)
        self.transform = transform
        self.use_crop = use_crop

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = str(self.dataframe.loc[idx, "image_name"])
        img_path = resolve_image_path(self.image_folder, img_name)
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image could not be read: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.use_crop:
            image = crop_black_borders(image)
        label = int(self.dataframe.loc[idx, "diagnosis"])
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label


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
        "efficientnet_b0", pretrained=False, num_classes=0,
        drop_rate=0.3, drop_path_rate=0.2
    )
    return EfficientNetWithHead(
        backbone=backbone, in_features=backbone.num_features,
        num_classes=5, dropout=0.5
    )


def apply_cam_mask_percentile(image_tensor, cam, top_percentage):
    image = image_tensor.clone()
    threshold_value = np.percentile(cam, 100 * (1 - top_percentage))
    mask_2d = cam >= threshold_value
    mask_tensor = torch.tensor(mask_2d, dtype=torch.bool, device=image.device)
    for c in range(image.shape[0]):
        image[c, mask_tensor] = 0.0
    return image

def calculate_fidelity_all(model, dataset, cam_obj, sampled_indices, top_percentages):
    drops = {p: [] for p in top_percentages}
    per_image = []
    for idx in tqdm(sampled_indices, desc="  Fidelity"):
        image_tensor, _ = dataset[idx]
        input_tensor = image_tensor.unsqueeze(0).to(device).float()
        with torch.no_grad():
            probs = torch.softmax(model(input_tensor), dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            original_confidence = probs[0, predicted_class].item()
        cam, _, _ = cam_obj.generate(input_tensor=input_tensor,
                                     class_idx=predicted_class, image_size=IMG_SIZE)
        row = {"index": idx}
        for p in top_percentages:
            masked = apply_cam_mask_percentile(image_tensor, cam, p)
            with torch.no_grad():
                mprobs = torch.softmax(model(masked.unsqueeze(0).to(device).float()), dim=1)
                masked_confidence = mprobs[0, predicted_class].item()
            drop = original_confidence - masked_confidence
            drops[p].append(drop)
            row[f"top{int(p*100)}"] = drop
        per_image.append(row)
    agg = {p: (float(np.mean(drops[p])), float(np.std(drops[p]))) for p in top_percentages}
    return agg, per_image


def add_small_noise(image_tensor, noise_std=0.015):
    return image_tensor + torch.randn_like(image_tensor) * noise_std

def calculate_stability(model, dataset, cam_obj, sampled_indices):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    similarities_corr = []
    similarities_ssim = []
    per_image = []

    for idx in tqdm(sampled_indices, desc="  Stability"):
        image_tensor, _ = dataset[idx]
        input_tensor = image_tensor.unsqueeze(0).to(device).float()

        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        cam_original, _, _ = cam_obj.generate(
            input_tensor=input_tensor, class_idx=predicted_class, image_size=IMG_SIZE
        )

        noisy_image = add_small_noise(image_tensor)
        noisy_tensor = noisy_image.unsqueeze(0).to(device).float()

        cam_noisy, _, _ = cam_obj.generate(
            input_tensor=noisy_tensor, class_idx=predicted_class, image_size=IMG_SIZE
        )

        corr = np.corrcoef(cam_original.flatten(), cam_noisy.flatten())[0, 1]
        if not np.isnan(corr):
            similarities_corr.append(corr)

        ssim_value = ssim(
            cam_original.astype(np.float32),
            cam_noisy.astype(np.float32),
            data_range=1.0
        )
        if not np.isnan(ssim_value):
            similarities_ssim.append(ssim_value)

        per_image.append({
            "index": idx,
            "corr": float(corr),
            "ssim": float(ssim_value),
        })

    agg = {
        "corr_mean": float(np.mean(similarities_corr)),
        "corr_std": float(np.std(similarities_corr)),
        "ssim_mean": float(np.mean(similarities_ssim)),
        "ssim_std": float(np.std(similarities_ssim)),
    }
    return agg, per_image

def main():
    print("AI_ROOT:", AI_ROOT)
    print("Device:", device)
    print("Image dir:", IMAGE_DIR)
    print("Model path:", MODEL_PATH)
    print("Test CSV:", TEST_CSV)
    print("Output dir:", EVAL_OUTPUT_DIR)
    print("Methods:", list(METHODS.keys()))

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")

    test_df = pd.read_csv(TEST_CSV)
    if "image_name" not in test_df.columns or "diagnosis" not in test_df.columns:
        raise ValueError("test.csv must contain columns: image_name, diagnosis")
    test_df["image_name"] = test_df["image_name"].astype(str)
    test_df["diagnosis"] = test_df["diagnosis"].astype(int)

    print("\nTest distribution:")
    print(test_df["diagnosis"].value_counts().sort_index())

    test_transforms = Compose([
        Resize(IMG_SIZE, IMG_SIZE),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    test_dataset = FundusDataset(
        dataframe=test_df, image_folder=IMAGE_DIR,
        transform=test_transforms, use_crop=True
    )

    model = build_model()
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    target_layer = model.backbone.blocks[TARGET_LAYER_INDEX]

    sample_count = min(MAX_SAMPLES, len(test_dataset))
    if sample_count >= len(test_dataset):
        sampled_indices = list(range(len(test_dataset)))
    else:
        sampled_indices = random.sample(range(len(test_dataset)), sample_count)

    results = {}
    all_fid_rows = []
    all_stab_rows = []

    for method_tag, MethodClass in METHODS.items():
        print(f"\n=== Metoda: {method_tag} ===")
        cam_obj = MethodClass(model=model, target_layer=target_layer, device=device)

        fid, fid_rows = calculate_fidelity_all(
            model, test_dataset, cam_obj, sampled_indices, TOP_PERCENTAGES
        )

        for r in fid_rows:
            r["method"] = method_tag

        all_fid_rows.extend(fid_rows)

        stab, stab_rows = calculate_stability(model, test_dataset, cam_obj, sampled_indices)

        for r in stab_rows:
            r["method"] = method_tag
        all_stab_rows.extend(stab_rows)

        cam_obj.remove_hooks()
        results[method_tag] = {"fidelity": fid, "stability": stab}

    fid_per_image_path = REPORTS_DIR / "fidelity_per_image_DDR_test.csv"
    pd.DataFrame(all_fid_rows).to_csv(fid_per_image_path, index=False)
    print("Saved fidelity per-image CSV:", fid_per_image_path)

    stab_per_image_path = REPORTS_DIR / "stability_per_image_DDR_test.csv"
    pd.DataFrame(all_stab_rows).to_csv(stab_per_image_path, index=False)
    print("Saved stability per-image CSV:", stab_per_image_path)
    
    comparison_rows = []
    for method_tag in METHODS.keys():
        r = results[method_tag]
        row = {"method": method_tag}
        for p in TOP_PERCENTAGES:
            m, s = r["fidelity"][p]
            row[f"fidelity_top{int(p*100)}_mean"] = round(m, 4)
            row[f"fidelity_top{int(p*100)}_std"] = round(s, 4)
        row["stab_corr_mean"] = round(r["stability"]["corr_mean"], 4)
        row["stab_corr_std"] = round(r["stability"]["corr_std"], 4)
        row["stab_ssim_mean"] = round(r["stability"]["ssim_mean"], 4)
        row["stab_ssim_std"] = round(r["stability"]["ssim_std"], 4)
        comparison_rows.append(row)

    df_cmp = pd.DataFrame(comparison_rows)
    df_cmp.to_csv(CSV_PATH, index=False)

    with open(TXT_PATH, "w", encoding="utf-8") as f:
        f.write("Fidelity & Stability - COMPARATIE Grad-CAM vs Grad-CAM++ vs X-Grad-CAM\n")
        f.write("Dataset: DDR | Split: test\n")
        f.write("Model: EfficientNet-B0 custom head\n")
        f.write(f"Model path: {MODEL_PATH}\n")
        f.write(f"Test CSV: {TEST_CSV}\n")
        f.write(f"Image dir: {IMAGE_DIR}\n")
        f.write(f"Samples: {sample_count}\n")
        f.write(f"Image size: {IMG_SIZE}x{IMG_SIZE}\n")
        f.write(f"Target layer: blocks[{TARGET_LAYER_INDEX}]\n\n")
        for method_tag in METHODS.keys():
            r = results[method_tag]
            f.write(f"===== {method_tag} =====\n")
            for p in TOP_PERCENTAGES:
                m, s = r["fidelity"][p]
                f.write(f"fidelity_top_{int(p*100)}_mean: {m:.4f}\n")
                f.write(f"fidelity_top_{int(p*100)}_std:  {s:.4f}\n")
            f.write(f"stability_corr_mean: {r['stability']['corr_mean']:.4f}\n")
            f.write(f"stability_corr_std:  {r['stability']['corr_std']:.4f}\n")
            f.write(f"stability_ssim_mean: {r['stability']['ssim_mean']:.4f}\n")
            f.write(f"stability_ssim_std:  {r['stability']['ssim_std']:.4f}\n\n")

    print("\n========== COMPARATIE (Fidelity = drop in incredere, mai mare = mai bine) ==========")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(df_cmp.to_string(index=False))
    print("\nSaved CSV:", CSV_PATH)
    print("Saved TXT:", TXT_PATH)


if __name__ == "__main__":
    main()