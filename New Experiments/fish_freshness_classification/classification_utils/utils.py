import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from data_preprocessing.image_processing import ImageProcessor
from data_preprocessing.extract_features import *
from data_preprocessing.color_spaces import ColorSpaces
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

def classification_metrics(model, img_arr, labels_arr, 
                           label_map, neural_net=False):
    
    predictions = model.predict(img_arr)
    if neural_net:
        predictions = np.array([np.argmax(pred) 
                                for pred in predictions])
    print(f"\nmodel predictions:\n{predictions}\n")

    # Generate confusion matrix
    cm = confusion_matrix(labels_arr, predictions)
    fish_type = list(label_map.keys())
    
    # Plot the confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', \
              xticklabels=fish_type, yticklabels=fish_type)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.show()

    test_species = [fish_type[label] for label in labels_arr]
    predicted_species = [fish_type[label] for label in predictions]
    # Calculate classification metrics
    report = classification_report(test_species, 
                                   predicted_species, 
                                   zero_division=1)
    print("\n",report)
    
def save_Model(model, model_path, neural_net=False):
    if not neural_net:
        # Save the model to a file
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        model.save(model_path)
        
def load_Model(model_path, neural_net=False):
    if not neural_net:
        # Load the model from the file
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        # load model
        model = load_model(model_path)
    return model



# DEEP LEARNING FEATURE EXTRACTION

def image_processing(img_arr, processing='edges'):
    new_img_array = []
    for image in img_arr:
        image_processor = ImageProcessor(image)
        if processing=='edges':
            # filter can be 'scahrr' or 'sobel'
            # img_type can be 'rgb' or 'bgr' or 'gray'
            edge_detected_img = image_processor.avgblur_edge_detection(
                                    filter='scahrr', img_type='gray')
            new_img_array.append(edge_detected_img)
            
        elif processing=='threshold':
            # threshold_type can be 'manual' or 'otsu'
            # manual has a threshold of 20
            binary_img = image_processor.image_thresholding(
                                    threshold_type='manual')
            new_img_array.append(binary_img)

        elif processing=='contours':
            # mode can be 'retr_external' or 'retr_list'
            contour = image_processor.contour_detection(mode='retr_external', 
                                                   draw_contours=False)
            new_img_array.append(contour)

        elif processing=='contour_crop':
            cropped_img = image_processor.max_contour_area_img_cropping()
            new_img_array.append(cropped_img)

        elif processing=='contrast':
            # final_img_colorspace can be 'hsv', 'rgb', 'grayscale'
            contrast_img = image_processor.contrast_image(final_img_colorspace='hsv')
            new_img_array.append(contrast_img)
            
        elif processing=='LBP_image':
            obj = LBP_features(image)
            # numPoints<<<radius is working for our usecase fish
            lbp_features, lbp_img = obj.extract_lbp_features(
                                                numPoints=5, radius=20)
            new_img_array.append(lbp_img)   
        
        elif processing=='saturation_image':
            saturation_img = image_processor.saturation_image()
            new_img_array.append(saturation_img)
    
    return new_img_array


def get_dl_data(bgr_images_arr, colorspace=False, color='hsv',
             img_processing=True, processing='edges'):
    c = ColorSpaces(bgr_images_arr)
    if colorspace:
        # convert colorspaces : color_space can be 'hsv', 'grayscale', 'rgb'
        converted_color_space_images = c.convert_color_space(color_space=color)
        normalized_image_array = c.normalize_images(
                                images=converted_color_space_images, 
                                                color_space=color)
    elif img_processing:
        new_img_array = image_processing(processing=processing)
        normalized_image_array = c.normalize_images(images=new_img_array,
                                                    color_space='grayscale')
    return normalized_image_array





# MACHINE LEARNING FEATURE EXTRACTION TECHNIQUES 
# NOTE : They all return normalized values

def extract_features(img_arr, technique='image_histograms'):
    if len(img_arr.shape) == 2:  # Grayscale image
        img_arr = img_arr.reshape(img_arr.shape[0], img_arr.shape[1], 1)
    elif len(img_arr.shape) == 3:  # Single image
        img_arr = img_arr.reshape(1, *img_arr.shape)
        
    features = []
    for image in img_arr:
        if technique=='image_histograms':
            obj = ImageHistograms(image)
            # colorspace can be 'hsv', 'rgb', 'bgr', 'gray'
            hist_features = obj.histograms(colorspace='hsv')
            features.append(hist_features)   
            
        elif technique=='HOG':
            obj = HOG_features(image)
            hog_features, hog_image = obj.extract_hog_features()
            features.append(hog_features)            

        elif technique=='LBP_histogram':
            obj = LBP_features(image)
            # numPoints<<<radius is working for our usecase fish
            lbp_features, lbp_img = obj.extract_lbp_features(
                                                numPoints=5, radius=20)
            features.append(lbp_features)   
            
        elif technique=='GLCM':
            obj = GLCM_features(image)
            glcm_features = obj.extract_glcm_features(
                            distances = [1, 2, 3],
                            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4])
            features.append(glcm_features) 
    
        elif technique=='saturation_hist':
            obj = Saturation_hist()
            saturation_hist = obj.extract_saturation_hist(image)
            features.append(saturation_hist)
    return np.array(features)


def beep(duration_ms=5000):
    import winsound
    duration = duration_ms  # milliseconds
    freq = 950  # Hz
    winsound.Beep(freq, duration)

