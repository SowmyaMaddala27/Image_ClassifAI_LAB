from data_preprocessing.data_loader import DataLoader
from evaluation.test_model import *
from training_models.models import *
from classification_utils.utils import *

# dataset_folder = r"D:\New folder\Dataset versions\sardine\train data"
dataset_folder = r"D:\New folder\Dataset versions\sardine\Blur classification Data"\
                r"\v2\train"
target_size = (280, 180) # width should be greater than height
# classes = ['Bad', 'Good']
classes = ['Blur', 'Clear']
label_map = {species: i for i, species in enumerate(classes)}
model_folder_path=r"D:\New folder\New Experiments\fish_freshness_classification"\
                    r"\training_models\blur_saved_models"
                
d = DataLoader(target_size, label_map, dataset_folder)

# fresh_images, fresh_labels = d.load_folder_images(folder_name="Good", 
#                                                   aspect_ratio=True)
# stale_images, stale_labels = d.load_folder_images(folder_name="Bad", 
#                                                   aspect_ratio=True)  
  
all_images, all_labels = d.load_all_folder_images(aspect_ratio=True)

img_data = get_dl_data(bgr_images_arr=all_images, colorspace=True, 
            color='rgb', img_processing=False, processing='edges')

# train val data splitting
train_images, val_images, train_labels, val_labels = d.split_data(
    img_data, all_labels, test_size=0.2, random_state=42)

if len(train_images.shape)==4: target_img_channels = 3
else: target_img_channels=1

# DL model training
def deep_learning(transfer_learning=False, 
                model_filepath=model_folder_path):
    obj = DL_model_training(num_classes=len(classes), 
                train_images=train_images, train_labels=train_labels, 
                val_images=val_images, val_labels=val_labels, epochs=10, 
                batch_size=32, target_img_width=target_size[0], 
                target_img_height=target_size[1], 
                target_img_channels=target_img_channels,  
                metrics=['accuracy'])
    if not transfer_learning:
        model = obj.cnn_model()
        model_filepath = model_filepath+r"\cnn_model.h5"
    else:
        model = obj.transfer_learning()
        model_filepath = model_filepath+r"\mobilenetv2_model.h5"
    trained_model = obj.model_training(model, model_filepath, 
                       early_stopping_patience=20,
                       use_data_augmentation=False, 
                       use_callbacks=True)
    return trained_model

# trained_model = deep_learning(transfer_learning=False)
trained_model = deep_learning(transfer_learning=True)


# model_filepath = model_folder_path+r"\rgb_93_mobilenetv2_model_v1.h5"
dl_model_filepath = model_folder_path+r"\mobilenetv2_model.h5"
neural_net=True
image_classifier = ImageClassifier(dl_model_filepath, target_size, 
                                   classes, neural_net)

# img_path = r"D:\New folder\Dataset versions\sardine\clear images data\train data\Good\20230523094649527_sardine_good.jpeg"
# image_classifier.test_new_data(img_path, visualize=False)

test_dataset_folder = r"D:\New folder\Dataset versions\sardine" \
                        r"\Blur classification Data\v2\test"
image_classifier.test_img_folder(test_dataset_folder)

image_classifier.test_img_folder(dataset_folder)









# MACHINE LEARNING MODELS

img_data = extract_features(img_arr=all_images, 
                            technique='saturation_hist')
test_img_data = extract_features(img_arr=test_images, 
                                 technique='saturation_hist')

# train val data splitting
train_images, val_images, train_labels, val_labels = d.split_data(
    img_data, all_labels, test_size=0.2, random_state=42)

neural_net=False

obj = ML_model_training(train_images, train_labels, 
                 val_images, val_labels)
models_dict = obj.ml_main()

trained_model = models_dict['svm']
model_path = model_folder_path+r"\saturation_hist_svm.pkl"
save_Model(model=trained_model, model_path=model_path, 
           neural_net=neural_net)


ml_model_filepath = model_folder_path+r"\saturation_hist_svm.pkl"
image_classifier = ImageClassifier(ml_model_filepath, target_size, 
                                   classes, neural_net)

# img_path = r"D:\New folder\Dataset versions\sardine\clear images data\train data\Good\20230523094649527_sardine_good.jpeg"
# image_classifier.test_new_data(img_path, visualize=False)

test_dataset_folder = r"D:\New folder\Dataset versions\sardine" \
                        r"\Blur classification Data\v2\test"
image_classifier.test_img_folder(test_dataset_folder)

image_classifier.test_img_folder(dataset_folder)