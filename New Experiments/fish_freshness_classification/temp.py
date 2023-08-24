from data_preprocessing.data_loader import DataLoader
from main import get_dl_data, deep_learning
from classification_utils.utils import *

dataset_folder = r"D:\New folder\Dataset versions\sardine\train data"
target_size = (280, 180) # width should be greater than height
classes = ['Bad', 'Good']
label_map = {species: i for i, species in enumerate(classes)}
model_filepath=r"D:\New folder\New Experiments\
                fish_freshness_classification\training_models\saved_models"
                
d = DataLoader(dataset_folder, target_size, label_map)
all_images, all_labels = d.load_all_folder_images(aspect_ratio=True)

test_dataset_folder = r"D:\New folder\Dataset versions\sardine\test data"
test_d = DataLoader(test_dataset_folder, target_size, label_map)
test_images, test_labels = test_d.load_all_folder_images(aspect_ratio=True)


img_data = get_dl_data(colorspace=False, color='hsv',
             img_processing=False, processing='edges',
             bgr_images_arr=all_images)

test_img_data = get_dl_data(colorspace=False, color='hsv',
             img_processing=False, processing='edges',
             bgr_images_arr=test_images)

# train val data splitting
train_images, val_images, train_labels, val_labels = d.split_data(
    img_data, all_labels, test_size=0.2, random_state=42)

if len(train_images.shape)==4: target_img_channels = 3
else: target_img_channels=1

trained_model = deep_learning(transfer_learning=True)

classification_metrics(model=trained_model, img_arr=test_img_data, 
                       labels_arr=test_labels, label_map=label_map, 
                       neural_net=True)
