import os
from PIL import Image
from torch.utils.data import Dataset

class SkinLesionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.image_to_mask = {img: self.get_mask_name(img) for img in self.images}

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = self.image_to_mask[img_name]
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        # Open the image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Apply transformations
        image, mask = self.transform(image, mask)

        # Convert mask to appropriate type if necessary
        # For segmentation tasks, masks often need to be of type LongTensor
        # Ensure mask has a channel dimension
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Shape becomes [1, H, W]

        # Convert mask to float tensor
        mask = mask.float()

        return image, mask

    def get_mask_name(self, img_name):
        if img_name.endswith('.jpg'):
            return img_name.replace('.jpg', '_segmentation.png')
        elif img_name.endswith('.png'):
            return img_name.replace('.png', '_segmentation.png')
        else:
            raise ValueError(f"Unexpected file extension in {img_name}")
        