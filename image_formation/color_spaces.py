import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

class ColorSpace:
    """
    A class to explore and visualize different color space representations of an image.
    """
    def __init__(self, img_path: str):
        """
        Initialize with the image path.

        Args:
            img_path (str): Path to the image file to be processed.
        """
        self.img_path = img_path

    def load_image(self, is_grayscale: bool) -> np.ndarray:
        """
        Load the image from the provided path, with optional grayscale conversion.

        Args:
            is_grayscale (bool): If True, load image in grayscale. Otherwise, load in color.

        Returns:
            img (np.ndarray): Loaded image in grayscale or BGR format.
        """
        # ***** Load the image using OpenCV *****
        self.img = cv.imread(self.img_path)  
        if is_grayscale:
            # ***** Convert to grayscale if requested *****
            self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)  
            return self.img
        return self.img

    def visualize_image(self,
                        image: np.ndarray,
                        image_gray: Optional[np.ndarray],
                        flag: str) -> None:
        """
        Visualize the individual color channels and grayscale version of the image.

        Args:
            image (np.ndarray): Image in the target color space (RGB/HSV/Lab).
            image_gray (Optional[np.ndarray]): Grayscale image, if available.
            flag (str): Color space identifier ('rgb', 'hsv', or 'lab').

        Returns:
            None
        """
        _, ax = plt.subplots(2, 3, figsize=(12, 6))
        flag = flag.lower()

        # ***** Set channel labels based on color space *****
        if flag == 'rgb':
            channel_names = ['Red', 'Green', 'Blue']
        elif flag == 'hsv':
            channel_names = ['Hue', 'Saturation', 'Value']
        else:
            channel_names = ['L* (Lightness)', 'a* (Green-Red)', 'b* (Blue-Yellow)']

        # ***** Plot each color channel *****
        for i in range(3):
            row, col = 0, i
            if flag == 'hsv' and i == 0: 
                ax[row, col].imshow(image[:, :, i], cmap='hsv') 
            else:
                ax[row, col].imshow(image[:, :, i], cmap='gray')
            ax[row, col].set_title(f'{channel_names[i]} Channel')
            ax[row, col].axis('off')

        # ***** Show the color image *****
        ax[1, 1].imshow(image)
        ax[1, 1].set_title(f'{flag.upper()} Image')
        ax[1, 1].axis('off')

        # ***** Show the grayscale image if available *****
        if image_gray is not None:
            ax[1, 0].imshow(image_gray, cmap='gray')
            ax[1, 0].set_title('Grayscale Image')
            ax[1, 0].axis('off')
        else:
            ax[1, 0].axis('off')

        # ***** Empty subplot for layout symmetry *****
        ax[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

    def explore_rgb(self, is_visualize: bool = True) -> None:
        """
        Convert the image to RGB and optionally visualize its channels and grayscale version.

        Args:
            is_visualize (bool): If True, visualizes the RGB image and channels.

        Returns:
            None
        """
        # ***** Load original BGR image *****
        img_bgr = self.load_image(False) 
        # ***** Load grayscale version *****
        img_gray = self.load_image(True)  
        # ***** Convert BGR to RGB *****
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)  

        if is_visualize:
            self.visualize_image(img_rgb, img_gray, 'rgb')

    def explore_hsv(self, is_visualize: bool = True) -> None:
        """
        Convert the image to HSV and optionally visualize its channels.

        Args:
            is_visualize (bool): If True, visualizes the HSV image and channels.

        Returns:
            None
        """
        # ***** Load original BGR image *****
        img_bgr = self.load_image(False)  
        # ***** Convert BGR to HSV *****
        img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)  

        if is_visualize:
            self.visualize_image(img_hsv, None, 'hsv')

    def explore_lab(self, is_visualization: bool = True) -> None:
        """
        Convert the image to Lab and optionally visualize its channels.

        Args:
            is_visualization (bool): If True, visualizes the Lab image and channels.

        Returns:
            None
        """
        # ***** Load original BGR image *****
        img_bgr = self.load_image(False)
        # ***** Convert BGR to Lab (fixed COLOR_RGB2Lab to COLOR_BGR2Lab) *****
        img_lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2Lab)  

        if is_visualization:
            self.visualize_image(img_lab, None, 'lab')


if __name__ == '__main__': 
    image = 'images/image.jpg'
    color_space = ColorSpace(image)
    color_space.explore_rgb()
    color_space.explore_hsv()
    color_space.explore_lab()