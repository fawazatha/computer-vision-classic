import matplotlib.pyplot as plt 
from matplotlib.axes import Axes
from matplotlib.widgets import Slider

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

def setup_cube_rotation_sliders(slider_axes: list[Axes]) -> tuple[Slider, Slider, Slider]:
    """
    Create sliders for rotating the cube along the Z, Y, and X axes.

    Args:
        slider_axes (List[Axes]): A list of Matplotlib Axes to place the sliders on. 
                                  Must have at least 3 axes.

    Returns:
        Tuple[Slider, Slider, Slider]: Sliders for cube rotation around Z, Y, and X axes respectively.
    """
    s_rz = Slider(slider_axes[0], 'Cube Rot Z', -180, 180, valinit=0)
    s_ry = Slider(slider_axes[1], 'Cube Rot Y', -180, 180, valinit=0)
    s_rx = Slider(slider_axes[2], 'Cube Rot X', -180, 180, valinit=0)
    return s_rz, s_ry, s_rx


def setup_camera_translation_sliders(slider_axes: list[Axes]) -> tuple[Slider, Slider, Slider]:
    """
    Create sliders for translating the camera along the Z, Y, and X axes.

    Args:
        slider_axes (List[Axes]): A list of Matplotlib Axes to place the sliders on.
                                  Must have at least 6 axes (index 3 to 5 used here).

    Returns:
        Tuple[Slider, Slider, Slider]: Sliders for camera translation along Z, Y, and X axes respectively.
    """
    s_tz = Slider(slider_axes[3], 'Cam Z', -10.0, 0.0, valinit=-3.0)
    s_ty = Slider(slider_axes[4], 'Cam Y', -5.0, 5.0, valinit=0.5)
    s_tx = Slider(slider_axes[5], 'Cam X', -5.0, 5.0, valinit=0.5)
    return s_tz, s_ty, s_tx


def setup_camera_rotation_sliders(slider_axes: list[Axes]) -> tuple[Slider, Slider, Slider]:
    """
    Create sliders for rotating the camera along the Z, Y, and X axes.

    Args:
        slider_axes (List[Axes]): A list of Matplotlib Axes to place the sliders on.
                                  Must have at least 9 axes (index 6 to 8 used here).

    Returns:
        Tuple[Slider, Slider, Slider]: Sliders for camera rotation around Z, Y, and X axes respectively.
    """
    s_crz = Slider(slider_axes[6], 'Cam Rot Z', -180, 180, valinit=0)
    s_cry = Slider(slider_axes[7], 'Cam Rot Y', -180, 180, valinit=0)
    s_crx = Slider(slider_axes[8], 'Cam Rot X', -180, 180, valinit=0)
    return s_crz, s_cry, s_crx


def setup_focal_length_sliders(slider_axes: list[Axes]) -> tuple[Slider, Slider]:
    """
    Create sliders for adjusting the camera's focal length in the Y and X directions.

    Args:
        slider_axes (List[Axes]): A list of Matplotlib Axes to place the sliders on.
                                  Must have at least 11 axes (index 9 and 10 used here).

    Returns:
        Tuple[Slider, Slider]: Sliders for focal length in Y and X directions respectively.
    """
    s_fy = Slider(slider_axes[9], 'Focal Y', 100, 500, valinit=250)
    s_fx = Slider(slider_axes[10], 'Focal X', 100, 500, valinit=250)
    return s_fy, s_fx