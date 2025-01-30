# main.py

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

sys.path.append('/scripts/')

from hair_removal import remove_hair_from_images
from models import save_model_weights
from SkinLesionDataset import SkinLesionDataset
from CrossValidationTrainer import CrossValidationTrainer
from CustomTransforms import CustomTransforms
from ModelTester import ModelTester


# Choose dataset type: 'with_hair' or 'without_hair'
dataset_type = 'without_hair'  # Options: 'with_hair', 'without_hair'

if dataset_type == 'with_hair':

    # Paths to images and masks
    images = '/datasets/HAM10000/images/'
    images_masks = '/datasets/HAM10000/masks/'

    # Paths to test images and masks
    test_images_= '/datasets/HAM10000/test_images/'
    test_masks = '/datasets/HAM10000/test_masks/'

    # Paths to save models and results
    best_models = '/app/skinLesions/segmentation/models/'
    training_results = '/app/skinLesions/segmentation/training_results/'
    test_results = '/app/skinLesions/segmentation/test_results/'
elif dataset_type == 'without_hair':
    # Paths to images and masks
    images = '/datasets/HAM10000/images_no_hair/'
    images_masks = '/datasets/HAM10000/masks/'

    # Paths to test images and masks
    test_images = '/datasets/HAM10000/test_images_no_hair/'
    test_masks = '/datasets/HAM10000/test_masks/'

    # Paths to save models and without hair
    best_models = '/app/skinLesions/segmentation/models_no_hair/'
    training_results = '/app/skinLesions/segmentation/training_results_no_hair/'
    test_results = '/app/skinLesions/segmentation/test_results_no_hair/'
else:
    raise ValueError("Invalid dataset_type. Choose 'with_hair' or 'without_hair'.")


# Remove hair from test images if needed
apply_hair_removal = False
remove_hair_from_images(test_images, test_images, apply_hair_removal)

# Choose model
model_name = 'unet_plus_plus'

if model_name == 'unet':
    from models import get_pretrained_unet
    model_fn = get_pretrained_unet
    model_dir = os.path.join(best_models, "unet")
    training_data_output_dir = os.path.join(training_results, "unet")
    testing_data_output_dir = os.path.join(test_results, "unet")
elif model_name == 'unet_plus_plus':
    from models import get_pretrained_unet_plus_plus
    model_fn = get_pretrained_unet_plus_plus
    model_dir = os.path.join(best_models, "unet_plus_plus")
    training_data_output_dir = os.path.join(training_results, "unet_plus_plus")
    testing_data_output_dir = os.path.join(test_results, "unet_plus_plus")
elif model_name == 'deep_lab':
    from models import get_pretrained_deep_lab
    model_fn = get_pretrained_deep_lab
    model_dir = os.path.join(best_models, "deep_lab")
    training_data_output_dir = os.path.join(training_results, "deep_lab")
    testing_data_output_dir = os.path.join(test_results, "deep_lab")

# Hyperparameters and settings
INPUT_SIZE = (224, 224)
BATCH_SIZE = 8
NUM_FOLDS = 5
MAX_EPOCHS = 10000
CRITERION = None

# Transformations for validation (and test) datasets
val_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),  # Resize to match the input size of the models
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Augmentations for the training dataset
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),          # 50% chance to apply horizontal flip
    transforms.RandomVerticalFlip(p=0.5),            # 50% chance to apply vertical flip
    transforms.RandomRotation(degrees=15),           # Rotate image by up to 15 degrees
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomResizedCrop(size=INPUT_SIZE, scale=(0.8, 1.0)),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=10
    ),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Instantiate the transform for training
train_transform = CustomTransforms(is_train=True)

# Create the dataset
train_val_dataset = SkinLesionDataset(
    image_dir=images,
    mask_dir=images_masks,
    transform=train_transform
)

# Cross-validation
trainer = CrossValidationTrainer(
    dataset=train_val_dataset,
    model_fn=model_fn,
    criterion=CRITERION,
    num_folds=NUM_FOLDS,
    batch_size=BATCH_SIZE,
    num_epochs=MAX_EPOCHS,
    lr=0.001,
    patience=10
)

results_df, best_model_weights = trainer.cross_validate()

# Save models and results as needed
best_model_paths = []

for fold_idx, model_weights in enumerate(best_model_weights):
    # Create a new model instance
    model = model_fn()

    # Load the best weights into the model
    model.load_state_dict(model_weights)

    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Save the model weights to a file
    model_save_path = os.path.join(model_dir, f"best_segmentation_fold{fold_idx + 1}.pth")
    save_model_weights(model, model_save_path)

    # Collect the model paths
    best_model_paths.append(model_save_path)


# Save training results
results_df.to_csv(os.path.join(training_data_output_dir, 'training.csv'), index=False)

