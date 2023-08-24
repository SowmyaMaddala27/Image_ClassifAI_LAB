import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, target_size, label_map, dataset_folder=''):
        self.dataset_folder = dataset_folder
        self.target_size = target_size
        self.label_map = label_map

    def preprocess_img(self, image, aspect_ratio=False):
        if image is not None:
            if aspect_ratio:
                height, width, channels = image.shape
                if width<height :
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)                
            preprocessed_image = cv2.resize(image, self.target_size)
            return preprocessed_image
        
    def load_folder_images(self, folder_name, aspect_ratio):
        images = []
        labels = []
        folder_path = os.path.join(self.dataset_folder, folder_name)
        if os.path.isdir(folder_path):
            for file in tqdm(os.listdir(folder_path), 
                             desc=f'Loading {folder_name} images'):
                filepath = os.path.join(folder_path, file)
                if os.path.isfile(filepath):
                    image = cv2.imread(filepath)
                    if image is not None:
                        # Preprocess image 
                        image = self.preprocess_img(image=image, 
                                                    aspect_ratio=aspect_ratio)
                        # Append image and label to lists
                        images.append(image)
                        labels.append(self.label_map[folder_name])
        images = np.array(images)
        labels = np.array(labels)
        return images, labels
    
    def load_all_folder_images(self, aspect_ratio):
        images = []
        labels = []
        for class_name in os.listdir(self.dataset_folder):
            class_label = self.label_map.get(class_name, -1)
            if class_label == -1: continue
            class_images, class_labels = self.load_folder_images(folder_name=class_name, 
                                                      aspect_ratio=aspect_ratio)
            images.append(class_images)
            labels.append(class_labels)
            
        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
        return images, labels

    def split_data(self, images, labels, test_size=0.2, random_state=42):
        train_images, val_images, train_labels, val_labels = train_test_split(
            images, labels, test_size=test_size, 
            random_state=random_state, stratify=labels
        )
        return train_images, val_images, train_labels, val_labels
    
# Example usage
if __name__ == "__main__":
    dataset_folder = "path_to_dataset"
    target_size = (280, 180)
    classes = ['Bad', 'Good']
    label_map = {species: i for i, species in enumerate(classes)}
    
    data_loader = DataLoader(dataset_folder, target_size, label_map)
    
    fresh_images, fresh_labels = data_loader.load_folder_images(folder_name="Good", 
                                                                aspect_ratio=True)
    stale_images, stale_labels = data_loader.load_folder_images(folder_name="Bad", 
                                                                aspect_ratio=True)    
    all_images, all_labels = data_loader.load_all_folder_images(aspect_ratio=True)
    
    train_images, val_images, train_labels, val_labels = data_loader.split_data(
        all_images, all_labels, test_size=0.2, random_state=42)
