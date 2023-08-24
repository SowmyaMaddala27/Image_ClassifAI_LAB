import cv2
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing.data_loader import DataLoader
from classification_utils.utils import *

class ImageClassifier:
    def __init__(self, model_path, target_size, classes, neural_net=True):
        self.model = load_Model(model_path=model_path, neural_net=neural_net)
        self.target_size = target_size
        self.label_map = {species: i for i, species in enumerate(classes)}
        self.neural_net = neural_net
        
    def preprocess_image(self, image):
        data_loader = DataLoader(self.target_size, self.label_map)
        preprocessed_img = data_loader.preprocess_img(image, aspect_ratio=True)
        if self.neural_net:
            preprocessed_img = get_dl_data(bgr_images_arr=preprocessed_img, 
                                           colorspace=True, color='bgr')
        else:
            preprocessed_img = extract_features(img_arr=preprocessed_img, 
                                                technique='saturation_hist')
        return preprocessed_img
    
    def predict_image(self, preprocessed_img):
        prediction = self.model.predict(preprocessed_img)
        if self.neural_net:
            prediction = np.array([np.argmax(pred) for pred in prediction])  
        reverse_label_map = {idx: img_type for img_type, idx in self.label_map.items()}
        predicted_class = reverse_label_map[prediction[0]] 
        return predicted_class
    
    def visualize_image(self, image_path):
        test_img = cv2.imread(image_path)
        if test_img is None:
            raise Exception("\nUnsupported file type\n")
        rgb_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        figsize = (10, 5)
        if rgb_img.shape[0] > rgb_img.shape[1]:
            figsize = (5, 10)
        plt.figure(figsize=figsize)
        plt.imshow(rgb_img)
        plt.axis("off")
        plt.tight_layout()
    
    def test_new_data(self, image_path, visualize=False):
        test_img = cv2.imread(image_path)
        preprocessed_img = self.preprocess_image(test_img)
        predicted_class = self.predict_image(preprocessed_img)
        print(f"Predicted class: {predicted_class}\n")
        if visualize: self.visualize_image(image_path)

    def test_img_folder(self, test_dataset_folder):
        data_loader = DataLoader(self.target_size, self.label_map, 
                                 test_dataset_folder)
        test_images, test_labels = data_loader.load_all_folder_images(
                                                            aspect_ratio=True)
        if self.neural_net:
            test_img_data = get_dl_data(bgr_images_arr=test_images, 
                                        colorspace=True, color='rgb', 
                                        img_processing=False, 
                                        processing='edges')
        else:
            test_img_data = extract_features(img_arr=test_images, 
                                             technique='saturation_hist')
        classification_metrics(model=self.model, img_arr=test_img_data, 
                               labels_arr=test_labels, label_map=self.label_map, 
                               neural_net=self.neural_net)

def main():
    model_filepath = r"D:\New folder\New Experiments\fish_freshness_classification\training_models\blur_saved_models\saturation_hist_svm.pkl"
    neural_net = False
    target_size = (280, 180)
    classes = ['Blur', 'Clear']
    
    image_classifier = ImageClassifier(model_filepath, target_size, 
                                       classes, neural_net)

    img_path = r"D:\New folder\Dataset versions\sardine\Blur classification Data\v2\test\Blur\20230526112103732_sardine_good.JPEG"
    image_classifier.test_new_data(img_path, visualize=False)

    test_dataset_folder = r"D:\New folder\Dataset versions\sardine\Blur classification Data\v2\test"
    image_classifier.test_img_folder(test_dataset_folder)

if __name__ == "__main__":
    main()
