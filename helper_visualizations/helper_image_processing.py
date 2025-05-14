import matplotlib 
import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider, RadioButtons

def sliders_similarity(similarity_x: float, slider_y_start: float, slider_step: float) -> list[Slider]:
    """
    Create sliders for similarity transformation parameters (Tx, Ty, Angle, Scale).

    Args:
        similarity_x (float): The x-position where the sliders should be placed.
        slider_y_start (float): The starting y-position for the first slider.
        slider_step (float): The vertical step size between each slider.

    Returns:
        List[Slider]: A list containing the created sliders for similarity transformation.
    """

    # ***** Create sliders for translation along x and y-axis, angle, and scale *****
    s_tx = Slider(plt.axes([similarity_x, slider_y_start, 0.35, 0.02]), 'Tx', -200, 200, valinit=0)
    s_ty = Slider(plt.axes([similarity_x, slider_y_start - slider_step, 0.35, 0.02]), 'Ty', -200, 200, valinit=0)
    s_angle = Slider(plt.axes([similarity_x, slider_y_start - 2*slider_step, 0.35, 0.02]), 'Angle', -180, 180, valinit=0)
    s_scale = Slider(plt.axes([similarity_x, slider_y_start - 3*slider_step, 0.35, 0.02]), 'Scale', 0.1, 3.0, valinit=1.0)
    
    # ***** Bundle sliders into a list *****
    sim_sliders = [s_tx, s_ty, s_angle, s_scale]
    
    return sim_sliders


def sliders_affine(affine_x: float, slider_y_start: float, slider_step: float) -> list[Slider]:
    """
    Create sliders for affine transformation parameters (a, b, c, d, Tx, Ty).

    Args:
        affine_x (float): The x-position where the sliders should be placed.
        slider_y_start (float): The starting y-position for the first slider.
        slider_step (float): The vertical step size between each slider.

    Returns:
        List[Slider]: A list containing the created sliders for affine transformation.
    """

    # ***** Create sliders for the affine transformation matrix parameters (a, b, c, d) *****
    s_a = Slider(plt.axes([affine_x, slider_y_start, 0.35, 0.02]), 'a', -2.0, 2.0, valinit=1.0)
    s_b = Slider(plt.axes([affine_x, slider_y_start - slider_step, 0.35, 0.02]), 'b', -2.0, 2.0, valinit=0.0)
    s_c = Slider(plt.axes([affine_x, slider_y_start - 2*slider_step, 0.35, 0.02]), 'c', -2.0, 2.0, valinit=0.0)
    s_d = Slider(plt.axes([affine_x, slider_y_start - 3*slider_step, 0.35, 0.02]), 'd', -2.0, 2.0, valinit=1.0)
    
    # ***** Create sliders for translation in affine transformation (Tx_affine, Ty_affine) *****
    s_tx_aff = Slider(plt.axes([affine_x, slider_y_start - 4*slider_step, 0.35, 0.02]), 'Tx_affine', -100, 100, valinit=0)
    s_ty_aff = Slider(plt.axes([affine_x, slider_y_start - 5*slider_step, 0.35, 0.02]), 'Ty_affine', -100, 100, valinit=0)
    
    # ***** Bundle sliders into a list *****
    aff_sliders = [s_a, s_b, s_c, s_d, s_tx_aff, s_ty_aff]
    
    return aff_sliders