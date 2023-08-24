import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, bgr_image):
        self.image = bgr_image

    def avgblur_edge_detection(self, filter='scharr', img_type='gray'):
        ksize = 15
        if img_type=='gray':
            self.grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.blur_image = cv2.blur(self.grayscale_image, (ksize, ksize))
        elif img_type=='rgb':
            self.rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.blur_image = cv2.blur(self.rgb_image, (ksize, ksize))
        elif img_type=='bgr':
            self.blur_image = cv2.blur(self.image, (ksize, ksize))
            
        if filter=='sobel':
            # Apply Sobel filters to detect edges in both X and Y directions
            sobel_x = cv2.Sobel(self.blur_image, cv2.CV_64F, 1, 0)
            sobel_y = cv2.Sobel(self.blur_image, cv2.CV_64F, 0, 1)
            self.edges = np.sqrt(sobel_x**2 + sobel_y**2).astype(np.uint8)
        elif filter=='scharr':
            # Apply Scharr filters to detect edges in both X and Y directions
            scharr_x = cv2.Scharr(self.blur_image, cv2.CV_64F, 1, 0)
            scharr_y = cv2.Scharr(self.blur_image, cv2.CV_64F, 0, 1)
            self.edges = np.sqrt(scharr_x**2 + scharr_y**2).astype(np.uint8)
        else:
            raise Exception("\nChoose sobel or scharr filter for edge detection\n")
        return self.edges
    

    def image_thresholding(self, threshold_type='manual'):
        edges = self.avgblur_edge_detection()
        if threshold_type=='manual':
            threshold = 20
            _, binary_image = cv2.threshold(edges, threshold, 255, 
                                            cv2.THRESH_BINARY)
        elif threshold_type=='otsu':
            _, binary_image = cv2.threshold(edges, 0, 255, cv2.THRESH_OTSU)
        else:
            raise Exception("\nChoose manual or otsu threshold for binary image\n")
        return binary_image
    
    def contour_detection(self, mode='retr_external', draw_contours=False):
        binary_image = self.image_thresholding()
        if mode=='retr_external':
            contours, _ = cv2.findContours(binary_image, 
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
        elif mode=='retr_list':
            contours, _ = cv2.findContours(binary_image, 
                                          cv2.RETR_LIST, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            # contours = sorted(contours, key=cv2.contourArea)[-1]
        else:
            raise Exception("\nChoose retr_external or retr_list mode\
                            for contour detection\n")
 
        if draw_contours:
            print(len(contours))
            # Create a blank image to draw the contours on
            contour_image = np.zeros_like(self.edges)
            # contour_image = self.grayscale_image.copy()
            # contour_image = self.image.copy()

            cv2.drawContours(contour_image, contours, -1, (255, 254, 0), 2);
            plt.imshow(contour_image, cmap='gray');    
        return contours
    
    def max_contour_area_img_cropping(self):
        print("\nUse scharr edge detection filter\n")
        contours = self.contour_detection(mode='retr_list')
        max_contour = sorted(contours, key=cv2.contourArea)[-1]
        x, y, w, h = cv2.boundingRect(max_contour)
        cropped_img = self.image[y:y + h, x:x + w]
        return cropped_img
    
    def contrast_image(self, final_img_colorspace='hsv'):
        blurred_image = cv2.GaussianBlur(self.image, (5,5), 0)
        hsv_img = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv_img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(v)
        final_img = cv2.merge((h,s,cl))
        if final_img_colorspace=='rgb':
            final_img = cv2.cvtColor(final_img, cv2.COLOR_HSV2RGB)
        elif final_img_colorspace=='grayscale':
            rgb_img = cv2.cvtColor(final_img, cv2.COLOR_HSV2RGB)
            final_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        return final_img
    
    def saturation_image(self):
        # use only saturation channel images
        converted_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        s_channel_img = converted_image[:,:,1]
        return s_channel_img
