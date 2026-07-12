import cv2
import timm
import torch
import base64
import numpy as np
from pathlib import Path
from skimage.segmentation import slic
import torch.nn as nn


def crop_black_borders(image, tolerance=7):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > tolerance

    if mask.sum() == 0:
        return image

    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1

    return image[y_min:y_max, x_min:x_max]


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

    return EfficientNetWithHead(
        backbone=backbone,
        in_features=backbone.num_features,
        num_classes=5,
        dropout=0.5
    )


class RetinaPredictor:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = 512

        self.class_names = [
            "No DR",
            "Mild",
            "Moderate",
            "Severe",
            "Proliferative DR"
        ]

        project_root = Path(__file__).resolve().parents[2]

        self.model_path = (
            project_root
            / "modelExperiments"
            / "DDR"
            / "outputs_ddr_effnetb0"
            / "models"
            / "best_effnetb0_ddr.pth"
        )

        self.model = build_model()

        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image_bytes):
        file_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image file.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = crop_black_borders(image, tolerance=7)

        image = cv2.resize(image, (self.image_size, self.image_size))

        original_image = image.astype(np.float32) / 255.0
        tensor = self.image_to_tensor(original_image)

        return tensor, original_image

    def image_to_tensor(self, image):
        image = image.astype(np.float32)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        normalized = (image - mean) / std

        tensor = torch.tensor(normalized, dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        return tensor.to(self.device)

    def get_confidence(self, image, target_class):
        tensor = self.image_to_tensor(image)

        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)[0]

        return probabilities[target_class].item()

    def normalize_cam(self, cam):
        cam = cam.astype(np.float32)
        cam = cv2.GaussianBlur(cam, (11, 11), 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def fidelity_percentile_drops(
        self,
        original_image,
        cam,
        target_class,
        original_confidence,
        top_percentages=(0.05, 0.10, 0.15, 0.20),
    ):
        base_tensor = self.image_to_tensor(original_image)

        drops = {}
        for p in top_percentages:
            threshold_value = np.percentile(cam, 100 * (1 - p))
            mask_2d = cam >= threshold_value
            mask_tensor = torch.tensor(mask_2d, dtype=torch.bool, device=base_tensor.device)

            masked = base_tensor.clone()
            for c in range(masked.shape[1]):
                masked[0, c, mask_tensor] = 0.0

            with torch.no_grad():
                probs = torch.softmax(self.model(masked), dim=1)[0]
            masked_conf = float(probs[target_class].item())

            drops[float(p)] = float(original_confidence - masked_conf)

        return drops

    def perturb_image(self, image):
        noise = np.random.normal(0, 0.015, image.shape)
        perturbed = np.clip(image + noise, 0, 1)
        return perturbed.astype(np.float32)

    def create_baseline_image(self, original_image, mode="blur"):
        if mode == "black":
            return np.zeros_like(original_image, dtype=np.float32)

        if mode == "mean":
            mean_color = original_image.mean(axis=(0, 1), keepdims=True)
            return np.ones_like(original_image, dtype=np.float32) * mean_color

        blurred = cv2.GaussianBlur(original_image, (31, 31), 0)
        return blurred.astype(np.float32)

    def get_auc_metrics(
        self,
        image,
        cam,
        target_class,
        n_segments=100,
        baseline_mode="blur"
    ):
        segments = slic(
            image,
            n_segments=n_segments,
            compactness=10,
            start_label=1,
            channel_axis=-1
        )

        segment_scores = []
        for label in np.unique(segments):
            mask = segments == label
            importance = float(np.mean(cam[mask]))
            segment_scores.append((label, importance))

        segment_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_labels = [label for label, _ in segment_scores]

        baseline = self.create_baseline_image(image, mode=baseline_mode)

        deletion_scores = []
        insertion_scores = []
        x_values = []

        total_segments = len(sorted_labels)

        for step in range(total_segments + 1):
            fraction = step / total_segments
            selected_labels = sorted_labels[:step]

            deletion_image = image.copy()
            insertion_image = baseline.copy()

            for label in selected_labels:
                mask = segments == label
                deletion_image[mask] = baseline[mask]
                insertion_image[mask] = image[mask]

            deletion_scores.append(self.get_confidence(deletion_image, target_class))
            insertion_scores.append(self.get_confidence(insertion_image, target_class))
            x_values.append(fraction)

        deletion_auc = float(np.clip(np.trapz(deletion_scores, x_values), 0.0, 1.0))
        insertion_auc = float(np.clip(np.trapz(insertion_scores, x_values), 0.0, 1.0))

        return deletion_auc, insertion_auc

    def create_heatmap_overlay(self, original_image, cam):
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_JET
        )

        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0

        overlay = 0.55 * original_image + 0.45 * heatmap
        overlay = np.clip(overlay, 0, 1)

        overlay_uint8 = np.uint8(overlay * 255)

        _, buffer = cv2.imencode(
            ".png",
            cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR)
        )

        return base64.b64encode(buffer).decode("utf-8")