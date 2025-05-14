# ***** IMPORT *****
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ***** IMPORT *****    
import numpy                as np
import cv2                  as cv
import matplotlib.pyplot    as plt
from matplotlib.widgets     import RadioButtons

# ***** IMPORT HELPER *****
from helper_visualizations.helper_image_processing import sliders_affine, sliders_similarity

class ImageWarping:
    """ 
    Class for applying image transformations using similarity and affine transformations, and visualizing the results.
    """
    def __init__(self, image_path: str,
                 sim_sliders: list,
                 aff_sliders: list,
                 ax_img,
                 center_radio: RadioButtons,
                 border_radio: RadioButtons,
                 mode_radio) -> None:
        """
        Initializes the ImageWarping object with image, sliders, and radio button configurations.

        Args:
            image_path (str): Path to the image file.
            sim_sliders (List[Slider]): Similarity transformation sliders.
            aff_sliders (List[Slider]): Affine transformation sliders.
            ax_img (Axes): Matplotlib axis to display the image.
            center_radio (RadioButtons): Radio buttons for center or corner selection.
            border_radio (RadioButtons): Radio buttons for border mode selection.
            mode_radio (RadioButtons): Radio buttons for transformation mode selection.
        """
        # ***** Read and convert the image from BGR to RGB *****
        self.image = cv.imread(image_path, cv.IMREAD_COLOR)
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
        
        # ***** Get image dimensions (height, width) *****
        self.height, self.width = self.image.shape[:2]

        # ***** Assign sliders, axis, and radio buttons to attributes *****
        self.sim_sliders = sim_sliders
        self.aff_sliders = aff_sliders
        self.ax_img = ax_img
        self.img_obj = ax_img.imshow(self.image)
        self.center_radio = center_radio
        self.border_radio = border_radio
        self.mode_radio = mode_radio

    def get_center(self) -> tuple:
        """
        Get the center point for transformations based on the selected anchor type.

        Args:
            None

        Returns:
            tuple: The (x, y) coordinates for the transformation center.
        """

        # ***** Determine center or corner based on radio button selection *****
        choice = self.center_radio.value_selected
        if choice == 'Center':
            return (self.width / 2, self.height / 2)
        elif choice == 'Corner':
            return (0, 0)
        
    def get_border_mode(self) -> tuple:
        """
        Get the border mode and value for transformations based on selected radio button.

        Args:
            None

        Returns:
            tuple: The border mode and border value for OpenCV warp functions.
        """

        # ***** Choose border handling mode based on radio button selection *****
        choice = self.border_radio.value_selected
        if choice == 'Constant (Blue)':
            return cv.BORDER_CONSTANT, (0, 0, 255)  # Blue color border
        elif choice == 'Replicate':
            return cv.BORDER_REPLICATE, None  # Replicate the edge pixels
        
    def get_similarity_transformed(self) -> np.ndarray:
        """
        Apply similarity transformation (translation, rotation, scaling) to the image.

        Args:
            None

        Returns:
            np.ndarray: The transformed image after similarity transformation.
        """
        
        # ***** Get translation, angle, and scale from sliders *****
        tx = self.sim_sliders[0].val
        ty = self.sim_sliders[1].val
        angle = self.sim_sliders[2].val
        scale = self.sim_sliders[3].val
        
        # ***** Calculate the rotation matrix with center, angle, and scale *****
        center = self.get_center()
        rot_mat = cv.getRotationMatrix2D(center, angle, scale)
       
        # ***** Apply translation by adjusting the rotation matrix *****
        rot_mat[0, 2] += tx  # Translate in x-direction
        rot_mat[1, 2] += ty  # Translate in y-direction

        # ***** Get border mode and value *****
        border_mode, border_value = self.get_border_mode()

        # ***** Warp the image using the affine transformation matrix *****
        transformed = cv.warpAffine(self.image,
                                   rot_mat,
                                   (self.width, self.height),
                                   flags=cv.INTER_LINEAR,
                                   borderMode=border_mode,
                                   borderValue=border_value if border_value is not None else 0)

        return transformed  
    
    def get_affine_transformed(self) -> np.ndarray:
        """
        Apply affine transformation (shear and translation) to the image.

        Args:
            None

        Returns:
            np.ndarray: The transformed image after affine transformation.
        """

        # ***** Get affine transformation parameters from sliders *****
        a = self.aff_sliders[0].val
        b = self.aff_sliders[1].val
        c = self.aff_sliders[2].val
        d = self.aff_sliders[3].val
        tx = self.aff_sliders[4].val
        ty = self.aff_sliders[5].val
        
        # ***** Construct the affine transformation matrix *****
        affine_matrix = np.array([[a, b, tx],
                                 [c, d, ty]], dtype=np.float32)

        # ***** Get border mode and value *****
        border_mode, border_value = self.get_border_mode()

        # ***** Warp the image using the affine matrix *****
        transformed = cv.warpAffine(self.image,
                                    affine_matrix,
                                    (self.width, self.height),
                                    flags=cv.INTER_LINEAR,
                                    borderMode=border_mode,
                                    borderValue=border_value if border_value is not None else 0)

        return transformed
    
    def update_visualization(self, val=None) -> None:
        """
        Update the image visualization based on the selected transformation mode.

        Args:
            val (optional): Slider value or radio button selection (used for update events).

        Returns:
            None
        """

        # ***** Apply the selected transformation based on mode *****
        if self.mode_radio.value_selected == 'Similarity':
            transformed = self.get_similarity_transformed()
        else:
            transformed = self.get_affine_transformed()

        # ***** Update the image object with the transformed image *****
        self.img_obj.set_data(transformed)
        plt.draw()

if __name__ == '__main__':
    image_path = 'images/image.jpg'

    # ***** Initialize transform parameters *****
    init_tx = 0
    init_ty = 0
    init_angle = 0
    init_scale = 1

    # ***** Setup Matplotlib figure for visualization *****
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(left=0.4, bottom=0.45, right=0.95)

    # ***** Set positions for sliders and radio buttons *****
    similarity_x = 0.05
    affine_x = 0.55
    slider_y_start = 0.25
    slider_step = 0.045

    # ***** Create sliders for similarity transformation *****
    sim_sliders = sliders_similarity(similarity_x, slider_y_start, slider_step)

    # ***** Create sliders for affine transformation *****
    aff_sliders = sliders_affine(affine_x, slider_y_start, slider_step)

    # ***** Create radio buttons for center/corner, border mode, and transformation mode *****
    ax_center = plt.axes([0.05, 0.75, 0.2, 0.1])
    center_radio = RadioButtons(ax_center, ('Center', 'Corner'), active=0)

    ax_border = plt.axes([0.05, 0.60, 0.25, 0.1])
    border_radio = RadioButtons(ax_border, ('Constant (Blue)', 'Replicate'), active=0)

    ax_mode = plt.axes([0.05, 0.45, 0.2, 0.1])
    mode_radio = RadioButtons(ax_mode, ('Similarity', 'Affine'), active=0)
    
    # ***** Add titles for different sections of the UI *****
    plt.text(0.05, 0.32, 'Similarity Transform Sliders', fontsize=12, weight='bold', transform=plt.gcf().transFigure)
    plt.text(0.55, 0.32, 'Affine Transform Sliders', fontsize=12, weight='bold', transform=plt.gcf().transFigure)
    plt.text(0.05, 0.88, 'Anchor Options (Only for Transform Similarity)', fontsize=9, weight='bold', transform=plt.gcf().transFigure)
    plt.text(0.05, 0.73, 'Border Mode', fontsize=9, weight='bold', transform=plt.gcf().transFigure)
    plt.text(0.05, 0.58, 'Transform Mode', fontsize=9, weight='bold', transform=plt.gcf().transFigure)

    # ***** Initialize ImageWarping object *****
    image_warping = ImageWarping(
        image_path, 
        sim_sliders, 
        aff_sliders, 
        ax, 
        center_radio, 
        border_radio, 
        mode_radio
    )
    image_warping.update_visualization()

    # ***** Connect update method to sliders and radio buttons *****
    for s in sim_sliders + aff_sliders:
        s.on_changed(image_warping.update_visualization)
    
    center_radio.on_clicked(image_warping.update_visualization)
    border_radio.on_clicked(image_warping.update_visualization)
    mode_radio.on_clicked(image_warping.update_visualization)

    plt.show()
