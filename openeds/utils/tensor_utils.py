import torch
import numpy as np

from sklearn.preprocessing import normalize


def cast_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    if img.ndim != 2:
        raise ValueError('image should be a valid 2d array')

    image = img.copy()
    image = normalize(image).astype(np.float32)

    image = image[:, :, np.newaxis]
    image = image.transpose(2, 0, 1)
    return torch.from_numpy(image).float()

