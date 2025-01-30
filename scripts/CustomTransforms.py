import random
import numbers
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

random.seed(42)
torch.manual_seed(42)

class CustomTransforms:
    def __init__(self, is_train=True):
        self.is_train = is_train

        # Define the base transformations
        self.resize = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Define the augmentations for training
        if self.is_train:
            self.augmentations = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                ),
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=10
                ),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
            ]
        else:
            self.augmentations = []

    def __call__(self, image, mask=None):
        # Resize image and mask
        image = self.resize(image)
        if mask is not None:
            mask = self.resize(mask)

        # Apply augmentations if in training mode
        if self.is_train:
            for aug in self.augmentations:
                if isinstance(aug, transforms.RandomHorizontalFlip):
                    if random.random() < aug.p:
                        image = TF.hflip(image)
                        if mask is not None:
                            mask = TF.hflip(mask)

                elif isinstance(aug, transforms.RandomVerticalFlip):
                    if random.random() < aug.p:
                        image = TF.vflip(image)
                        if mask is not None:
                            mask = TF.vflip(mask)

                elif isinstance(aug, transforms.RandomRotation):
                    # Handle degrees
                    if isinstance(aug.degrees, numbers.Number):
                        degrees = [-aug.degrees, aug.degrees]
                    else:
                        degrees = [min(aug.degrees), max(aug.degrees)]
                    angle = random.uniform(degrees[0], degrees[1])
                    image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                    if mask is not None:
                        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

                elif isinstance(aug, transforms.ColorJitter):
                    image = aug(image)
                    # Do not apply color jitter to the mask

                elif isinstance(aug, transforms.RandomResizedCrop):
                    i, j, h, w = aug.get_params(
                        image, scale=aug.scale, ratio=aug.ratio
                    )
                    image = TF.resized_crop(
                        image, i, j, h, w, aug.size, interpolation=InterpolationMode.BILINEAR
                    )
                    if mask is not None:
                        mask = TF.resized_crop(
                            mask, i, j, h, w, aug.size, interpolation=InterpolationMode.NEAREST
                        )

                elif isinstance(aug, transforms.RandomAffine):
                    # Handle degrees
                    if isinstance(aug.degrees, numbers.Number):
                        degrees = [-aug.degrees, aug.degrees]
                    else:
                        degrees = [min(aug.degrees), max(aug.degrees)]
                    # Handle shear
                    if isinstance(aug.shear, numbers.Number):
                        shear = [-aug.shear, aug.shear]
                    elif isinstance(aug.shear, (list, tuple)):
                        if len(aug.shear) == 2:
                            shear = [min(aug.shear), max(aug.shear)]
                        elif len(aug.shear) == 4:
                            shear = aug.shear
                        else:
                            raise ValueError(
                                "Shear must be a number or a tuple/list of 2 or 4 values."
                            )
                    else:
                        shear = [0.0, 0.0]
                    params = aug.get_params(
                        degrees=degrees,
                        translate=aug.translate,
                        scale_ranges=aug.scale,
                        shears=shear,
                        img_size=image.size
                    )
                    image = TF.affine(
                        image, *params, interpolation=InterpolationMode.BILINEAR
                    )
                    if mask is not None:
                        mask = TF.affine(
                            mask, *params, interpolation=InterpolationMode.NEAREST
                        )

                elif isinstance(aug, transforms.RandomPerspective):
                    if random.random() < aug.p:
                        # Correct the argument order
                        width, height = image.size
                        startpoints, endpoints = aug.get_params(
                            width, height, aug.distortion_scale
                        )
                        image = TF.perspective(
                            image, startpoints, endpoints, interpolation=InterpolationMode.BILINEAR
                        )
                        if mask is not None:
                            mask = TF.perspective(
                                mask, startpoints, endpoints, interpolation=InterpolationMode.NEAREST
                            )

        # Convert to tensor
        image = TF.to_tensor(image)
        if mask is not None:
            mask = TF.to_tensor(mask)

        # Normalize image
        image = self.normalize(image)

        if mask is not None:
            return image, mask
        else:
            return image