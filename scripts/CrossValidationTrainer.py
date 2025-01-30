import torch
import random
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def dice_loss(pred, target, smooth=1e-6):
    # Ensure pred and target are float tensors
    pred = pred.float()
    target = target.float()

    # Ensure pred and target have the same shape
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred shape {pred.shape}, target shape {target.shape}")

    # Apply sigmoid activation to predictions if not already applied
    pred = torch.sigmoid(pred)

    # Calculate Dice loss
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice_score = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice_score.mean()

    return dice_loss

def iou_score(pred, target, smooth=1e-6):
    # Ensure pred and target are float tensors
    pred = pred.float()
    target = target.float()

    # Ensure pred and target have the same shape
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch in iou_score: pred shape {pred.shape}, target shape {target.shape}")

    # Apply sigmoid activation to predictions if not already applied
    pred = torch.sigmoid(pred)

    # Calculate intersection and union
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection

    # Compute IoU score
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

class CrossValidationTrainer:
    def __init__(self, dataset, model_fn, criterion=None, num_folds=5, batch_size=16, num_epochs=20, lr=0.001, patience=10):
        if criterion is None:
            criterion = dice_loss
        
        self.dataset = dataset
        self.model_fn = model_fn 
        self.criterion = criterion
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def train_fold(self, train_loader, val_loader):
        model = self.model_fn()
        model.to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        patience_counter = 0
        best_val_loss = float('inf')
        best_model_state = None

        epoch_data = {
            'train_loss': [],
            'train_iou': [],
            'val_loss': [],
            'val_iou': [],
        }

        for epoch in range(self.num_epochs):
            print(f'\nEpoch {epoch+1}/{self.num_epochs}')
            model.train()
            train_loss_sum = 0.0
            train_iou_sum = 0.0
            for batch in tqdm(train_loader, desc="Training"):
                images, masks = batch
                images = images.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, masks)
                iou = iou_score(outputs, masks)

                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                train_iou_sum += iou.item()

            train_loss = train_loss_sum / len(train_loader)
            train_iou = train_iou_sum / len(train_loader)
            epoch_data['train_loss'].append(train_loss)
            epoch_data['train_iou'].append(train_iou)

            print(f"Epoch {epoch+1} Training: Loss: {train_loss:.4f}, IoU: {train_iou:.4f}")

            # Validation step
            model.eval()
            val_loss_sum = 0.0
            val_iou_sum = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating"):
                    images, masks = batch
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    outputs = model(images)
                    loss = self.criterion(outputs, masks)
                    iou = iou_score(outputs, masks)

                    val_loss_sum += loss.item()
                    val_iou_sum += iou.item()

            val_loss = val_loss_sum / len(val_loader)
            val_iou = val_iou_sum / len(val_loader)
            epoch_data['val_loss'].append(val_loss)
            epoch_data['val_iou'].append(val_iou)

            print(f"Epoch {epoch+1} Validation: Loss: {val_loss:.4f}, IoU: {val_iou:.4f}")

            # Update the best model if the validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
                print('Best model updated!')
            else:
                patience_counter += 1

            # Adjust learning rate and optionally log it
            scheduler.step(val_loss)
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}')

            # Early stopping
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1} with best validation loss: {best_val_loss:.4f}")
                break

        return epoch_data, best_model_state

    def cross_validate(self):
        kfold = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        fold_results = []
        best_models = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            print(f"FOLD {fold+1}/{self.num_folds}")
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)

            epoch_data, best_model_state = self.train_fold(train_loader, val_loader)
            epoch_data['fold'] = fold + 1 

            fold_results.append(pd.DataFrame(epoch_data))
            best_models.append(best_model_state)

        # Combine all fold results into a single DataFrame
        results_df = pd.concat(fold_results, ignore_index=True)
        return results_df, best_models
