import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv

class Steel_dataset(Dataset):

    def __init__(self, image_dir, mask_dir, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg','.npy'))
        image = cv.imread(img_path)
        mask = np.load(mask_path)
        mask[mask==255.0]=1.0

        if self.transforms is not None:
            augmentations = self.transforms(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        
        return image, mask