import cv2
import numpy as np

class ColorSpaces:
    def __init__(self, images):
        if len(images.shape) == 2:  # Grayscale image
            images = images.reshape(images.shape[0], images.shape[1], 1)
        elif len(images.shape) == 3:  # Single image
            images = images.reshape(1, *images.shape)
        self.images = images

    def convert_color_space(self, color_space):
        converted_images = []
        for image in self.images:
            if color_space == 'hsv':
                converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif color_space == 'grayscale':
                converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif color_space == 'rgb':
                converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                converted_image = image  # No conversion for RGB or grayscale
            converted_images.append(converted_image)
        return np.array(converted_images)
    
    def normalize_images(self, images, color_space='rgb'):
        normalized_images = []
        # images = images.astype(np.float32)  # Convert to float32
        for image in images:
            if color_space == 'hsv':
                image[..., 0] = image[..., 0] / 179.0  # Normalize hue
                image[..., 1] = image[..., 1] / 255.0  # Normalize saturation
                image[..., 2] = image[..., 2] / 255.0  # Normalize value
            else:
                image = image / 255.0  # Normalize RGB or grayscale
            normalized_images.append(image)
        return np.array(normalized_images)

# Example usage
if __name__ == "__main__":
    color_space = 'hsv'  # Choose the desired color space

    # Example single image, grayscale image, and array of images
    single_image = cv2.imread("path_to_image.jpg")
    grayscale_image = cv2.imread("grayscale_image.jpg", cv2.IMREAD_GRAYSCALE)
    image_array = np.array([cv2.imread("image1.jpg"), cv2.imread("image2.jpg")])

    color_space_obj_single = ColorSpaces(single_image, color_space)
    color_space_obj_grayscale = ColorSpaces(grayscale_image, color_space)
    color_space_obj_array = ColorSpaces(image_array, color_space)

    # Convert color space and normalize images
    converted_single_image = color_space_obj_single.convert_color_space()
    converted_grayscale_image = color_space_obj_grayscale.convert_color_space()
    converted_image_array = color_space_obj_array.convert_color_space()

    normalized_single_image = color_space_obj_single.normalize_images(converted_single_image)
    normalized_grayscale_image = color_space_obj_grayscale.normalize_images(converted_grayscale_image)
    normalized_image_array = color_space_obj_array.normalize_images(converted_image_array)

    # Now you can proceed with other preprocessing steps
