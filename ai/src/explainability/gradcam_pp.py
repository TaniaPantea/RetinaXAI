import cv2
import torch
import numpy as np


class GradCAMPlusPlus:

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
                "Hook-urile nu au capturat gradienti sau activari! "
                "Verifica daca target_layer este corect."
            )

        grads = self.gradients[0]
        activations = self.activations[0]

        grads_2 = grads ** 2
        grads_3 = grads_2 * grads

        sum_activations = activations.sum(dim=(1, 2), keepdim=True)

        denom = 2.0 * grads_2 + sum_activations * grads_3
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
        alphas = grads_2 / denom

        weights = (alphas * torch.relu(grads)).sum(dim=(1, 2))

        cam = (weights.view(-1, 1, 1) * activations).sum(dim=0)

        cam = torch.relu(cam)

        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (image_size, image_size))

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        max_coords = np.unravel_index(np.argmax(cam, axis=None), cam.shape)

        if threshold is not None:
            cam = (cam > threshold).astype(np.float32)

        return cam, class_idx, max_coords