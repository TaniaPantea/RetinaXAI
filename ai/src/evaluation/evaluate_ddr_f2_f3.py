import cv2
import timm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score
)

import albumentations as A
from albumentations.pytorch import ToTensorV2



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 512
BATCH_SIZE = 16
NUM_WORKERS = 0

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

AI_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = AI_ROOT / "data" / "DR_grading"
IMAGE_DIR = DATASET_DIR

OUTPUT_DIR = AI_ROOT / "modelExperiments" / "DDR"

MODEL_PATH = OUTPUT_DIR / "outputs_ddr_effnetb0" / "models" / "best_effnetb0_ddr.pth"
TEST_CSV = OUTPUT_DIR / "outputs_ddr_effnetb0" / "splits" / "test.csv"

REPORTS_DIR = OUTPUT_DIR / "reports"
CM_DIR = OUTPUT_DIR / "confusion_matrices"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CM_DIR.mkdir(parents=True, exist_ok=True)



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

    raise FileNotFoundError(f"Image not found: {image_name} in {image_folder}")


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
        img_name = self.dataframe.loc[idx, "image_name"]
        img_path = resolve_image_path(self.image_folder, img_name)

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

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



def evaluate_model(model, loader):
    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating DDR test set"):
            images = images.to(device).float()

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    return np.array(y_true), np.array(y_pred)


def save_confusion_matrix(cm):
    cm_path = CM_DIR / "confusion_matrix_test_f2_f3_DDR.png"

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix - EfficientNetB0 DDR - Test")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()

    return cm_path



def main():
    print("Device:", device)
    print("AI_ROOT:", AI_ROOT)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("MODEL_PATH:", MODEL_PATH)
    print("TEST_CSV:", TEST_CSV)
    print("IMAGE_DIR:", IMAGE_DIR)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    test_df = pd.read_csv(TEST_CSV)

    print("\nTest CSV columns:", test_df.columns.tolist())
    print(test_df.head())

    if "image_name" not in test_df.columns or "diagnosis" not in test_df.columns:
        raise ValueError("test.csv must contain columns: image_name, diagnosis")

    test_df["image_name"] = test_df["image_name"].astype(str)
    test_df["diagnosis"] = test_df["diagnosis"].astype(int)

    print("\nTest distribution:")
    print(test_df["diagnosis"].value_counts().sort_index())

    test_transforms = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])

    test_dataset = FundusDataset(
        dataframe=test_df,
        image_folder=IMAGE_DIR,
        transform=test_transforms,
        use_crop=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    model = build_model()

    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    y_true, y_pred = evaluate_model(model, test_loader)

    accuracy = accuracy_score(y_true, y_pred)

    f2_macro = fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0)
    f2_weighted = fbeta_score(y_true, y_pred, beta=2, average="weighted", zero_division=0)

    f3_macro = fbeta_score(y_true, y_pred, beta=3, average="macro", zero_division=0)
    f3_weighted = fbeta_score(y_true, y_pred, beta=3, average="weighted", zero_division=0)

    f2_per_class = fbeta_score(y_true, y_pred, beta=2, average=None, zero_division=0)
    f3_per_class = fbeta_score(y_true, y_pred, beta=3, average=None, zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES)))
    )

    print("\n========== DDR TEST CLASSIFICATION METRICS ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F2 Macro: {f2_macro:.4f}")
    print(f"F2 Weighted: {f2_weighted:.4f}")
    print(f"F3 Macro: {f3_macro:.4f}")
    print(f"F3 Weighted: {f3_weighted:.4f}")

    print("\nPer-class F2 / F3:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: F2={f2_per_class[i]:.4f}, F3={f3_per_class[i]:.4f}")

    print("\nClassification report:")
    print(report)

    print("\nConfusion matrix:")
    print(cm)

    metrics_path = REPORTS_DIR / "classification_f2_f3_metrics_DDR_test.txt"
    csv_path = REPORTS_DIR / "classification_f2_f3_per_class_DDR_test.csv"

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Classification Metrics - DDR Test Set\n")
        f.write("Model: EfficientNet-B0 custom head\n")
        f.write(f"Model path: {MODEL_PATH}\n")
        f.write(f"Test CSV: {TEST_CSV}\n")
        f.write(f"Image dir: {IMAGE_DIR}\n")
        f.write(f"Image size: {IMG_SIZE}x{IMG_SIZE}\n\n")

        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F2 Macro: {f2_macro:.4f}\n")
        f.write(f"F2 Weighted: {f2_weighted:.4f}\n")
        f.write(f"F3 Macro: {f3_macro:.4f}\n")
        f.write(f"F3 Weighted: {f3_weighted:.4f}\n\n")

        f.write("Per-class F2 / F3:\n")
        for i, class_name in enumerate(CLASS_NAMES):
            f.write(f"{class_name}: F2={f2_per_class[i]:.4f}, F3={f3_per_class[i]:.4f}\n")

        f.write("\nClassification report:\n")
        f.write(report)

        f.write("\nConfusion matrix:\n")
        f.write(str(cm))

    df_results = pd.DataFrame({
        "class": CLASS_NAMES,
        "f2_score": f2_per_class,
        "f3_score": f3_per_class
    })

    df_results.to_csv(csv_path, index=False)

    cm_path = save_confusion_matrix(cm)

    print("\nSaved TXT:", metrics_path)
    print("Saved CSV:", csv_path)
    print("Saved confusion matrix:", cm_path)


if __name__ == "__main__":
    main()