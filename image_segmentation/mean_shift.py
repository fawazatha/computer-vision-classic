import cv2 as cv 
import numpy as np 
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt 

class MeanShiftSegment: 
    """
    Perform MeanShift-based image segmentation by combining color and spatial features.
    
    Reads an image, converts to RGB and Lab color spaces, constructs a feature vector
    combining Lab channels and pixel coordinates (scaled), runs MeanShift clustering,
    and recolors each segment with its mean RGB color.

    Args:
        img_path (str): Path to the input image file.
        h_color (float): Bandwidth scaling factor for Lab color channels.
        h_spatial (float): Bandwidth scaling factor for spatial (x,y) coordinates.
        quantile (float): Quantile parameter for estimating MeanShift bandwidth.
        n_samples (int): Number of samples to use when estimating bandwidth.
    """
    def __init__(self, 
                 img_path: str,
                 h_color: float, 
                 h_spatial: float, 
                 quantile: float,
                 n_samples: int) -> None:
        # ***** Load image from disk and validate *****
        self.open_img = cv.imread(img_path)
        if self.open_img is None:
            raise ValueError(f"Error: Could not open or find the image at {img_path}")

        # ***** Convert BGR image to RGB and Lab (float32) color spaces *****
        self.img_rgb = cv.cvtColor(self.open_img, cv.COLOR_BGR2RGB)
        self.img_lab = cv.cvtColor(self.open_img, cv.COLOR_BGR2LAB).astype(np.float32)

        self.quantile = quantile
        self.n_samples = n_samples
        self.h_color = h_color
        self.h_spatial = h_spatial

        # Placeholder for segmented output
        self.mean_color_segmented_image: np.ndarray | None = None
    
    def generate_meanshift(self) -> None:
        """
        Run MeanShift clustering on combined color-spatial features and build a segmented image.

        Returns:
            None: Results stored in self.mean_color_segmented_image.
        """
         # ***** Get image dimensions and flatten Lab color data *****
        height, width = self.img_lab.shape[:2]
        flat_image_lab = self.img_lab.reshape((-1, 3))

        # ***** Build coordinate grid and flatten *****
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        flat_coordinates = np.column_stack([x.flatten(), y.flatten()])

        # ***** Combine color and coordinates into a single feature array *****
        flat_img_with_coordinates = np.column_stack([flat_image_lab, flat_coordinates])

        # ***** Prepare array to hold scaled features *****
        scaled_features = np.zeros_like(flat_img_with_coordinates, dtype=np.float32)

        # ***** Scale Lab color channels by h_color *****
        if self.h_color > 0:
            scaled_features[:, 0:3] = flat_image_lab / self.h_color
        else:
            raise ValueError("h_color must be greater than 0 to avoid division by zero.")

        # ***** Scale spatial coordinates by h_spatial *****
        if self.h_spatial > 0:
            scaled_features[:, 3] = flat_coordinates[:, 0] / self.h_spatial  # x coordinate
            scaled_features[:, 4] = flat_coordinates[:, 1] / self.h_spatial  # y coordinate
        else:
            raise ValueError("h_spatial must be greater than 0 to avoid division by zero.")

        # ***** Estimate bandwidth for MeanShift clustering *****
        bandwidth = estimate_bandwidth(
            scaled_features,
            quantile=self.quantile,
            n_samples=self.n_samples
        )
        mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)

        # ***** Fit MeanShift to scaled features *****
        mean_shift.fit(scaled_features)

        # ***** Extract cluster labels and reshape to image grid *****
        labels = mean_shift.labels_
        segmented_image = labels.reshape((height, width))
        unique_labels = np.unique(labels)

        # ***** Prepare output image array *****
        self.mean_color_segmented_image = np.zeros_like(self.img_rgb, dtype=np.uint8)

        # ***** Compute mean RGB color for each cluster and assign *****
        for label_val in unique_labels:
            mask = (segmented_image == label_val)
            # ***** Compute mean color of original RGB pixels for this segment *****
            mean_color = np.mean(self.img_rgb[mask], axis=0).astype(np.uint8)
            # ***** Apply mean color to all pixels in this segment *****
            self.mean_color_segmented_image[mask] = mean_color
    
    def plot_meanshift(self) -> None:
        """
        Generate the MeanShift segmentation and display original vs segmented images side by side.

        Returns:
            None
        """
        # ***** Run the segmentation if not already done *****
        try: 
            self.generate_meanshift()
        except ValueError as error: 
            print(f"Error generating mean shift image: {error}")
            return
        
        if self.mean_color_segmented_image is None:
            print("MeanShift segmentation has not been generated yet. Please run generate_meanshift() first.")
            return

        # ***** Create matplotlib figure with two subplots *****
        fig, axes = plt.subplots(1, 2, figsize=(12, 7))

        # ***** Display original RGB image *****
        axes[0].imshow(self.img_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # ***** Display mean-color segmented image *****
        axes[1].imshow(self.mean_color_segmented_image)
        axes[1].set_title('MeanShift Segmented Image')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()
        

if __name__ == '__main__': 
    img_path = 'images/image.jpg'
    h_color_val = 20.0
    h_spatial_val=50.0
    meanshift_segment = MeanShiftSegment(img_path, h_color_val, h_spatial_val, 0.1, 500)
    meanshift_segment.plot_meanshift()        