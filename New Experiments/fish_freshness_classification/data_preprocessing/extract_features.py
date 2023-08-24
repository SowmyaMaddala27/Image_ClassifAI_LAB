# import the necessary packages
from skimage import feature
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model, Model
import cv2
from skimage.feature import hog, graycomatrix, graycoprops
from skimage import exposure
from data_preprocessing.image_processing import ImageProcessor

# '''
class ImageHistograms:
    def __init__(self, bgr_image):
        self.image = bgr_image
    
    def histograms(self, colorspace):
        if colorspace=='gray':
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist /= hist.sum()
            return hist
        
        elif (colorspace=='rgb') or (colorspace=='bgr'):
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

            # Calculate the RGB color histogram
            hist_red = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
            hist_green = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
            hist_blue = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])

            # Normalize the histograms
            hist_red /= hist_red.sum()
            hist_green /= hist_green.sum()
            hist_blue /= hist_blue.sum()

            # Concatenate the histograms into a feature vector
            hist_features = np.concatenate((hist_red, hist_green, hist_blue))
            hist_features = hist_features.squeeze()
            return hist_features
        
        elif colorspace=='hsv':
            # Convert the image from BGR to HSV
            image_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

            # Calculate the HSV color histogram
            hist_hue = cv2.calcHist([image_hsv], [0], None, [180], 
                                    [0, 180])
            hist_saturation = cv2.calcHist([image_hsv], [1], None, [256], 
                                           [0, 256])
            hist_value = cv2.calcHist([image_hsv], [2], None, [256], 
                                      [0, 256])

            # Normalize the histograms
            hist_hue /= hist_hue.sum()
            hist_saturation /= hist_saturation.sum()
            hist_value /= hist_value.sum()

            # Concatenate the histograms into a feature vector
            hist_features_hsv = np.concatenate((hist_hue, hist_saturation, 
                                                hist_value))
            hist_features_hsv = hist_features_hsv.squeeze()
            return hist_features_hsv
        
        
class HOG_features:
    def __init__(self, bgr_image):
        self.image = bgr_image
    
    def extract_hog_features(self, orientations=9, 
                                    pixels_per_cell=(16, 16), 
                                    cells_per_block=(4, 4)):
        
        obj = ImageProcessor(self.image)
        scharr_edges = obj.avgblur_edge_detection(filter='scharr', 
                                                  img_type='rgb')
        # Compute HOG features
        hog_features, hog_image = hog(scharr_edges, orientations=orientations, 
                                      pixels_per_cell=pixels_per_cell, 
                                      cells_per_block=cells_per_block, 
                                      visualize=True,
                                      channel_axis=-1)
        
        # Enhance the contrast of the HOG image
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))  
        return hog_features, hog_image_rescaled  


class LBP_features:
    def __init__(self, bgr_image):
        self.image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    
    def extract_lbp_features(self, numPoints=5, radius=20):
        """
		compute the Local Binary Pattern representation
		of the image, and then use the LBP representation
		to build the histogram of patterns
        """
        # numPoints<<<radius is working for our usecase fish
        lbp = feature.local_binary_pattern(self.image, 
                                           numPoints,
			                               radius, 
                                           method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
			                bins=np.arange(0, numPoints + 3),
			                range=(0, numPoints + 2))  
        # normalize the histogram
        eps=1e-7
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns 
        return hist, lbp
    
class GLCM_features:
    def __init__(self, bgr_image):
        self.image = bgr_image
    
    def extract_glcm_features(self, distances = [1, 2, 3],
                              angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        glcm_features = []
        for d in distances:
            for a in angles:
                glcm = graycomatrix(gray_image, [d], [a], 
                                    symmetric=True, normed=True)
                features = [
                    graycoprops(glcm, 'contrast')[0, 0],
                    graycoprops(glcm, 'energy')[0, 0],
                    graycoprops(glcm, 'correlation')[0, 0],
                    graycoprops(glcm, 'homogeneity')[0, 0],
                    graycoprops(glcm, 'dissimilarity')[0, 0]
                ]
                glcm_features.extend(features)
        return glcm_features
  
class Saturation_hist:
    def __init__(self):
        pass    
    def extract_saturation_hist(self, bgr_image):
        # use only saturation channel images
        converted_image = cv2.cvtColor(bgr_image, 
                                           cv2.COLOR_BGR2HSV)
        s_channel = converted_image[:,:,1]
        s_hist = cv2.calcHist([s_channel], [0], None, [256], [0, 256])
        s_hist /= s_hist.sum()
        s_hist = s_hist.squeeze()
        return s_hist

class SIFT_features:
    def __init__(self):
        pass
    
    def extract_sift_features(self):
        pass
    
class ORB_features:
    def __init__(self):
        pass
    
    def extract_orb_features(self):
        pass
    
class CNN_features:
    def __init__(self, bgr_image):
        self.image = bgr_image
        
    def cnn_feature_exractor(self, cnn_model_path, img_preprocessor):
        cnn_model = load_model(cnn_model_path)
        # Create an intermediate model with the flattened layer's output
        feature_extractor_model = Model(inputs=cnn_model.input, 
                                        outputs=cnn_model.layers[-3].output)

        # Extract features from a new image
        preprocessed_img = img_preprocessor(self.image)
        cnn_feature_map = feature_extractor_model.predict(preprocessed_img)
        return cnn_feature_map
# '''