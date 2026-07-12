import cv2
import torch
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None

        self.forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None, image_size=512, threshold=None):
        self.model.eval()

        self.gradients = None
        self.activations = None

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[:, class_idx]

        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Hook-urile nu au capturat gradienți sau activări! "
                "Verifică dacă target_layer este corect."
            )

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(self.device)
        for i, weight in enumerate(weights):
            cam += weight * activations[i]

        cam = torch.relu(cam)

        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (image_size, image_size))

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        max_coords = np.unravel_index(np.argmax(cam, axis=None), cam.shape)

        if threshold is not None:
            cam = (cam > threshold).astype(np.float32)

        return cam, class_idx, max_coords