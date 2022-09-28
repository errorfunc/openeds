import cv2

from albumentations import (
    Compose,
    OneOf,
    VerticalFlip,
    HorizontalFlip,
    ShiftScaleRotate,
    Blur,
    GaussianBlur,
    RandomBrightnessContrast,
    CLAHE,
    ImageCompression,
    MultiplicativeNoise,
    GaussNoise,
    Posterize,
    Solarize,
    GazeVectorParams,
)


def gaze_aug(p: float) -> 'Compose':
    return Compose([
        OneOf([
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.9),
            ShiftScaleRotate(
                shift_limit=0.2,
                rotate_limit=25,
                scale_limit=0.2,
                border_mode=cv2.BORDER_CONSTANT,
                interpolation=cv2.INTER_LANCZOS4,
                p=0.8
            )
        ], p=0.9),
        OneOf([
            Blur(blur_limit=5, p=0.7),
            GaussianBlur(blur_limit=7, p=0.7),
            RandomBrightnessContrast(brightness_limit=(-0.2, 0.4), contrast_limit=(-0.2, 0.3), p=0.8)
        ], p=0.7),
        OneOf([
            CLAHE(clip_limit=2, p=0.8),
            ImageCompression(quality_lower=20, quality_upper=55, p=0.8),
            MultiplicativeNoise(multiplier=(0.8, 1.5), elementwise=True, p=0.8),
            GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.8),
            Posterize(num_bits=4, p=0.8),
            Solarize(p=0.8)
        ], p=0.8),
    ],
        gaze_vector_params=GazeVectorParams(format='xyz'),
        p=p
    )
