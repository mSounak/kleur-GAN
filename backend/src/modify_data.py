from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import shutil



def split_train_val(folder_path, split=0.8):

    images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    train_images = images[:int(len(images) * 0.8)]
    val_images = images[int(len(images) * 0.8):]

    if not (os.path.exists(os.path.join(folder_path, 'train')) and os.path.exists(os.path.join(folder_path, 'val'))):
        os.makedirs(os.path.join(folder_path, 'train'))
        os.makedirs(os.path.join(folder_path, 'val'))

        for image in tqdm(train_images, desc='Copying train images'):
            shutil.copy(os.path.join(folder_path, image), os.path.join(folder_path, 'train'))
        for image in tqdm(val_images, desc='Copying val images'):
            shutil.copy(os.path.join(folder_path, image), os.path.join(folder_path, 'val'))

    if (os.path.exists(os.path.join(folder_path, 'train')) and os.path.exists(os.path.join(folder_path, 'val'))):
        for image in tqdm(images, desc='deleting images'):
            os.remove(os.path.join(folder_path, image))


def rename_data(folder_path):

    train_images = [f for f in os.listdir(os.path.join(folder_path, 'train')) if f.endswith('.jpg')]
    val_images = [f for f in os.listdir(os.path.join(folder_path, 'val')) if f.endswith('.jpg')]

    for i, image in enumerate(tqdm(train_images, desc='Renaming train images')):
        os.rename(os.path.join(folder_path, 'train', image), os.path.join(folder_path, 'train', "train_" + str(i) + '.jpg'))
    
    for i, image in enumerate(tqdm(val_images, desc='Renaming val images')):
        os.rename(os.path.join(folder_path, 'val', image), os.path.join(folder_path, 'val', "val_" + str(i) + '.jpg'))


if __name__ == "__main__":
    folder_path = 'data/images/'
    split_train_val(folder_path)
    rename_data(folder_path)