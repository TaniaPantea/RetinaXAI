import cv2
import timm
import torch
import numpy as np
import pandas as pd
import sys

from pathlib import Path
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset
from skimage.segmentation import slic
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
IMG_SIZE = 512
MAX_SAMPLES = 100
N_SEGMENTS = 200
BASELINE_MODE = "blur"

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DDR_DIR = AI_ROOT / "modelExperiments" / "DDR"

DATASET_DIR = AI_ROOT / "data" / "DR_grading"
IMAGE_DIR = DATASET_DIR

MODEL_PATH = DDR_DIR / "outputs_ddr_effnetb0" / "models" / "best_effnetb0_ddr.pth"
TEST_CSV = DDR_DIR / "outputs_ddr_effnetb0" / "splits" / "test.csv"

EVAL_OUTPUT_DIR = DDR_DIR / "outputs_slic_auc_eval"
REPORTS_DIR = EVAL_OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = REPORTS_DIR / "slic_insertion_deletion_comparison_DDR_test.csv"
DETAILED_CSV = REPORTS_DIR / "slic_insertion_deletion_detailed_DDR_test.csv"
TXT_PATH = REPORTS_DIR / "slic_insertion_deletion_comparison_DDR_test.txt"

METHODS = {
    "gradcam": GradCAMSimple,
    "gradcampp": GradCAMPlusPlus,
    "xgradcam": XGradCAM,
}

TARGET_LAYER_INDEX = 6

_trapz = getattr(np, "trapezoid", getattr(np, "trapz"))


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


def denormalize_tensor(image_tensor):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image.astype(np.float32)


def image_to_tensor_from_numpy(image):
    image = image.astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (image - mean) / std
    tensor = torch.tensor(normalized, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def get_confidence_numpy(model, image, target_class):
    tensor = image_to_tensor_from_numpy(image)
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]
    return probabilities[target_class].item()


def create_baseline_image(image, mode="blur"):
    if mode == "black":
        return np.zeros_like(image, dtype=np.float32)
    if mode == "mean":
        mean_color = image.mean(axis=(0, 1), keepdims=True)
        return np.ones_like(image, dtype=np.float32) * mean_color
    if mode == "blur":
        return cv2.GaussianBlur(image, (31, 31), 0).astype(np.float32)
    return np.zeros_like(image, dtype=np.float32)


def calculate_slic_auc(method_tag, model, dataset, cam_obj):
    model.eval()
    deletion_aucs = []
    insertion_aucs = []
    details = []

    sample_count = min(MAX_SAMPLES, len(dataset))

    for idx in tqdm(range(sample_count), desc=f"  {method_tag} SLIC AUC"):
        image_tensor, true_label = dataset[idx]
        input_tensor = image_tensor.unsqueeze(0).to(device).float()

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            original_confidence = probabilities[predicted_class].item()

        cam, _, _ = cam_obj.generate(
            input_tensor=input_tensor, class_idx=predicted_class, image_size=IMG_SIZE
        )
        cam = cam.astype(np.float32)
        cam = cv2.GaussianBlur(cam, (7, 7), 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        image = denormalize_tensor(image_tensor)
        baseline = create_baseline_image(image, mode=BASELINE_MODE)

        segments = slic(image, n_segments=N_SEGMENTS, compactness=10,
                        start_label=1, channel_axis=-1)

        segment_scores = []
        for label in np.unique(segments):
            mask = segments == label
            importance = float(np.mean(cam[mask]))
            segment_scores.append((label, importance))
        segment_scores.sort(key=lambda x: x[1], reverse=True)

        sorted_labels = [label for label, _ in segment_scores]
        total_segments = len(sorted_labels)

        deletion_scores, insertion_scores, x_values = [], [], []

        for step in range(total_segments + 1):
            fraction = step / total_segments
            selected_labels = sorted_labels[:step]

            deletion_image = image.copy()
            insertion_image = baseline.copy()
            for label in selected_labels:
                mask = segments == label
                deletion_image[mask] = baseline[mask]
                insertion_image[mask] = image[mask]

            deletion_conf = get_confidence_numpy(model, deletion_image, predicted_class)
            insertion_conf = get_confidence_numpy(model, insertion_image, predicted_class)

            deletion_scores.append(deletion_conf)
            insertion_scores.append(insertion_conf)
            x_values.append(fraction)

        deletion_norm = np.clip(np.array(deletion_scores) / (original_confidence + 1e-8), 0, 1)
        insertion_norm = np.clip(np.array(insertion_scores) / (original_confidence + 1e-8), 0, 1)

        deletion_auc = float(_trapz(deletion_norm, x_values))
        insertion_auc = float(_trapz(insertion_norm, x_values))

        deletion_aucs.append(deletion_auc)
        insertion_aucs.append(insertion_auc)

        details.append({
            "method": method_tag,
            "index": idx,
            "true_label": int(true_label),
            "true_label_name": CLASS_NAMES[int(true_label)],
            "predicted_class": int(predicted_class),
            "predicted_class_name": CLASS_NAMES[int(predicted_class)],
            "original_confidence": float(original_confidence),
            "deletion_auc": deletion_auc,
            "insertion_auc": insertion_auc,
            "actual_segments": int(total_segments),
        })

    return {
        "deletion_mean": float(np.mean(deletion_aucs)),
        "deletion_std": float(np.std(deletion_aucs)),
        "insertion_mean": float(np.mean(insertion_aucs)),
        "insertion_std": float(np.std(insertion_aucs)),
        "details": details,
    }


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

    results = {}
    all_details = []
    for method_tag, MethodClass in METHODS.items():
        print(f"\n=== Metoda: {method_tag} ===")
        cam_obj = MethodClass(model=model, target_layer=target_layer, device=device)
        res = calculate_slic_auc(method_tag, model, test_dataset, cam_obj)
        cam_obj.remove_hooks()
        results[method_tag] = res
        all_details.extend(res["details"])

    pd.DataFrame(all_details).to_csv(DETAILED_CSV, index=False)

    summary_rows = []
    sample_count = min(MAX_SAMPLES, len(test_dataset))
    for method_tag in METHODS.keys():
        r = results[method_tag]
        summary_rows.append({
            "method": method_tag,
            "n_samples": sample_count,
            "deletion_mean": round(r["deletion_mean"], 4),
            "deletion_std": round(r["deletion_std"], 4),
            "insertion_mean": round(r["insertion_mean"], 4),
            "insertion_std": round(r["insertion_std"], 4),
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(SUMMARY_CSV, index=False)

    with open(TXT_PATH, "w", encoding="utf-8") as f:
        f.write("SLIC Insertion / Deletion AUC - COMPARATIE Grad-CAM vs Grad-CAM++ vs X-Grad-CAM\n")
        f.write("Dataset: DDR | Split: test\n")
        f.write("Model: EfficientNet-B0 custom head\n")
        f.write(f"Model path: {MODEL_PATH}\n")
        f.write(f"Test CSV: {TEST_CSV}\n")
        f.write(f"Image dir: {IMAGE_DIR}\n")
        f.write(f"Max Samples: {sample_count}\n")
        f.write(f"Image size: {IMG_SIZE}x{IMG_SIZE}\n")
        f.write(f"n_segments: {N_SEGMENTS}\n")
        f.write(f"Baseline: {BASELINE_MODE}\n")
        f.write(f"Target layer: blocks[{TARGET_LAYER_INDEX}]\n\n")
        f.write("Reminder: Deletion AUC mai MIC = mai bine; Insertion AUC mai MARE = mai bine.\n\n")
        for method_tag in METHODS.keys():
            r = results[method_tag]
            f.write(f"===== {method_tag} =====\n")
            f.write(f"Deletion AUC Mean:  {r['deletion_mean']:.4f}\n")
            f.write(f"Deletion AUC Std:   {r['deletion_std']:.4f}\n")
            f.write(f"Insertion AUC Mean: {r['insertion_mean']:.4f}\n")
            f.write(f"Insertion AUC Std:  {r['insertion_std']:.4f}\n\n")

    print("\n========== COMPARATIE SLIC AUC ==========")
    print("(Deletion mai mic = mai bine ; Insertion mai mare = mai bine)")
    print(df_summary.to_string(index=False))
    print("\nSaved summary CSV:", SUMMARY_CSV)
    print("Saved detailed CSV:", DETAILED_CSV)
    print("Saved TXT:", TXT_PATH)


if __name__ == "__main__":
    main()