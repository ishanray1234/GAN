from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import config

class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(root_dir)
        print(self.list_files)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.list_files[idx])
        image = np.array(Image.open(img_path))

        input_image = image[:, :600, :]  # Left half
        target_image = image[:, 600:, :]  # Right half

        augmentation = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentation['image']
        target_image = augmentation['image0']

        input_image = config.transform_only_input(image=input_image)['image']
        target_image = config.transform_only_mask(image=target_image)['image']
        return image