# ***** IMPORT *****
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib

matplotlib.use('TkAgg')  

class CubeProjection:
    """
    Handles the 3D-to-2D projection of a cube using camera intrinsics, extrinsics, 
    and user-controlled sliders for rotation and translation.
    """
    def __init__(self, 
                 cube_world: np.ndarray, 
                 cube_edges: list, 
                 ax, cx: float, cy: float, 
                 sliders: tuple):
        """
        Initialize CubeProjection.
        
        Args:
            cube_world: (8, 3) array of cube vertices in world space.
            cube_edges: list of tuples defining edge connections.
            cx, cy: principal point coordinates.
            ax: matplotlib axes for drawing.
            sliders: tuple of 11 sliders for camera, cube controls and focal length.
        """
        self.cube_world = cube_world 
        self.cube_edges = cube_edges
        self.ax = ax
        self.cx = cx 
        self.cy = cy 
        (self.s_tx, self.s_ty, self.s_tz, 
        self.s_rx, self.s_ry, self.s_rz, 
        self.s_crx, self.s_cry, self.s_crz,  
        self.s_fy, self.s_fx) = sliders

    def get_extrinsics(self, s_tx: Slider, s_ty: Slider, s_tz: Slider) -> np.ndarray:
        """
        Get the camera extrinsic matrix as a 4x4 homogeneous transformation matrix.

        Args:
            s_tx, s_ty, s_tz: sliders representing camera translation.

        Returns:
            extrinsic_4x4: (4, 4) homogeneous transformation matrix.
        """
        # ***** Camera position in world space *****
        C_w = np.array([s_tx.val, s_ty.val, s_tz.val])
        
        # ***** Camera rotation from sliders (in degrees → radians) *****
        angle_x = np.deg2rad(self.s_crx.val)
        angle_y = np.deg2rad(self.s_cry.val)
        angle_z = np.deg2rad(self.s_crz.val)
    
        # ***** Build rotation matrix for camera (R_c_w) *****
        R_c_w = (
            self.rotation_matrix_z(angle_z) @
            self.rotation_matrix_y(angle_y) @
            self.rotation_matrix_x(angle_x)
        )

        # ***** Convert to world-to-camera (i.e. inverse of R_c_w) *****
        R = R_c_w.T
        t = -R @ C_w

        # ***** Pack into 4x4 extrinsic matrix *****
        extrinsic_4x4 = np.eye(4)
        extrinsic_4x4[:3, :3] = R
        extrinsic_4x4[:3, 3] = t

        return extrinsic_4x4


    def get_intrinsic_matrix(self, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
        """
        Constructs the intrinsic camera matrix K using focal lengths and principal point.

        Args:
            fx (float): Focal length in x direction.
            fy (float): Focal length in y direction.
            cx (float): Principal point x-coordinate.
            cy (float): Principal point y-coordinate.

        Returns:
            np.ndarray: (3, 3) intrinsic camera matrix
        """
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    def translation_matrix(self, tx: float, ty: float, tz: float) -> np.ndarray:
        """
        Creates a 4x4 translation matrix for translating 3D coordinates.

        Args:
            tx (float): Translation along X-axis.
            ty (float): Translation along Y-axis.
            tz (float): Translation along Z-axis.

        Returns:
            np.ndarray: (4, 4) translation matrix.
        """
        return np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])

    def rotation_matrix_x(self, angle_rad: float) -> np.ndarray:
        """
        Generates a rotation matrix for rotating about the X-axis.

        Args:
            angle_rad (float): Rotation angle in radians.

        Returns:
            np.ndarray: (3, 3) rotation matrix around X-axis.
        """
        # ***** Build X-axis rotation matrix *****
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])

    def rotation_matrix_y(self, angle_rad: float) -> np.ndarray:
        """
        Generates a rotation matrix for rotating about the Y-axis.

        Args:
            angle_rad (float): Rotation angle in radians.

        Returns:
            np.ndarray: (3, 3) rotation matrix around Y-axis.
        """
        # ***** Build Y-axis rotation matrix *****
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

    def rotation_matrix_z(self, angle_rad: float) -> np.ndarray:
        """
        Generates a rotation matrix for rotating about the Z-axis.

        Args:
            angle_rad (float): Rotation angle in radians.

        Returns:
            np.ndarray: (3, 3) rotation matrix around Z-axis.
        """
        # ***** Build Z-axis rotation matrix *****
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

    def scale_matrix(self, sx: float, sy: float, sz: float) -> np.ndarray:
        """
        Generates a scale matrix to scale coordinates in 3D space.

        Args:
            sx (float): Scale along X-axis.
            sy (float): Scale along Y-axis.
            sz (float): Scale along Z-axis.

        Returns:
            np.ndarray: (4, 4) scale matrix.
        """
        return np.array([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ])
    
    def apply_transform(self, cube_vertices_homogeneous: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Applies a 4x4 transformation matrix to a set of homogeneous 3D cube vertices.

        Args:
            cube_vertices_homogeneous (np.ndarray): (N, 4) homogeneous coordinates of cube vertices.
            transform_matrix (np.ndarray): (4, 4) transformation matrix.

        Returns:
            np.ndarray: (N, 4) transformed homogeneous vertices.
        """
        return cube_vertices_homogeneous @ transform_matrix.T  

    def get_projection(self, p_world: np.ndarray, extrinsics: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        Projects 3D world-space points to 2D image-space coordinates using camera matrices.

        Args:
            p_world (np.ndarray): (N, 3) array of 3D points in world coordinates.
            extrinsics (np.ndarray): (4, 4) world-to-camera transform.
            K (np.ndarray): (3, 3) intrinsic camera matrix.

        Returns:
            np.ndarray: (N, 2) projected 2D image points.
        """
        # ***** Convert to homogeneous coordinates (N, 4) *****
        num_points = p_world.shape[0]
        p_world_hom = np.hstack([p_world, np.ones((num_points, 1))])  # (N, 4)

        # ***** Transform to camera coordinates (N, 4) *****
        p_camera_hom = p_world_hom @ extrinsics.T  # (N, 4)

        # ***** Drop homogeneous component to get (N, 3) *****
        p_camera = p_camera_hom[:, :3]

        # ***** Project to image plane using intrinsics *****
        p_image_hom = (K @ p_camera.T).T  # (N, 3)

        # ***** Normalize to get 2D image coordinates *****
        x_pixel = p_image_hom[:, 0] / p_image_hom[:, 2]
        y_pixel = p_image_hom[:, 1] / p_image_hom[:, 2]

        return np.stack([x_pixel, y_pixel], axis=1)  # (N, 2)

    def update_visualization(self, val: float | None):
        """
        Updates the cube projection visualization when any slider is moved.

        Args:
            val (float | None): Value from the slider callback (unused).
        """
        # ***** Compute projection matrix *****
        K = self.get_intrinsic_matrix(self.s_fx.val, self.s_fy.val, self.cx, self.cy)
        extrinsics = self.get_extrinsics(self.s_tx, self.s_ty, self.s_tz)

        # ***** Apply rotation to cube based on sliders *****
        angle_x = np.deg2rad(self.s_rx.val)
        angle_y = np.deg2rad(self.s_ry.val)
        angle_z = np.deg2rad(self.s_rz.val)

        R_cube = self.rotation_matrix_x(angle_x) @ self.rotation_matrix_y(angle_y) @ self.rotation_matrix_z(angle_z)
        R_cube_hom = np.eye(4)
        R_cube_hom[:3, :3] = R_cube
        # ***** Translate cube to center before rotation *****
        T_cube = self.translation_matrix(-0.5, -0.5, -0.5)
        transform = T_cube @ R_cube_hom

        # ***** Homogenize vertices and apply transform *****
        cube_vertices_hom = np.hstack([self.cube_world, np.ones((8, 1))])
        transformed_cube = self.apply_transform(cube_vertices_hom, transform)[:, :3]

        # ***** Project points and draw *****
        projected_pts = self.get_projection(transformed_cube, extrinsics, K)
        self.draw_wireframe(projected_pts)

    def draw_wireframe(self, projected_points: np.ndarray):
        """
        Draws the cube's wireframe on the Matplotlib axes using the projected 2D points.

        Args:
            projected_points (np.ndarray): (8, 2) array of 2D image-space coordinates.
        """
        # ***** Clear and draw lines between connected vertices *****
        self.ax.clear()
        self.ax.set_xlim(0, 640)
        self.ax.set_ylim(480, 0)
        for start, end in self.cube_edges:
            x_vals = [projected_points[start, 0], projected_points[end, 0]]
            y_vals = [projected_points[start, 1], projected_points[end, 1]]
            self.ax.plot(x_vals, y_vals, 'b-')
        self.ax.set_aspect('equal')
        self.ax.set_title('Projected Cube')


if __name__ == '__main__':
    # ***** Cube Definition *****
    cube_vertices_world = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
    ])
    cube_edges = [
        (0,1), (1,2), (2,3), (3,0),  # Bottom
        (4,5), (5,6), (6,7), (7,4),  # Top
        (0,4), (1,5), (2,6), (3,7)   # Sides
    ]

    # ***** Camera intrinsics *****
    cx, cy = 320, 240

    # ***** Setup figure *****
    fig, ax = plt.subplots(figsize=(8, 9))  # Main plot
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95)

    # ***** Slider layout config *****
    slider_params = {
        'left': 0.25,
        'width': 0.65,
        'height': 0.03,
        'spacing_v': 0.005,
        'bottom_margin': 0.02,
        'top_margin': 0.03,
    }
    
    num_sliders = 11
    slider_total_height = (
        num_sliders * (slider_params['height'] + slider_params['spacing_v']) +
        slider_params['top_margin'] +
        slider_params['bottom_margin']
    )

    # ***** Set bottom of main plot area so it doesn't overlap sliders *****
    plt.subplots_adjust(bottom=slider_total_height)

    # ***** Calculate Y positions (top-down visually, so reversed index) *****
    slider_y_bottoms = [
        slider_params['bottom_margin'] + i * (slider_params['height'] + slider_params['spacing_v'])
        for i in range(num_sliders)
    ]

    # ***** Slider axes from bottom up *****
    slider_axes = []
    for i in range(num_sliders):
        y = slider_y_bottoms[i]
        ax_slider = plt.axes([slider_params['left'], y, slider_params['width'], slider_params['height']])
        slider_axes.append(ax_slider)

    # ***** Setup sliders *****
    s_rz = Slider(slider_axes[0], 'Cube Rot Z', -180, 180, valinit=0)
    s_ry = Slider(slider_axes[1], 'Cube Rot Y', -180, 180, valinit=0)
    s_rx = Slider(slider_axes[2], 'Cube Rot X', -180, 180, valinit=0)

    s_tz = Slider(slider_axes[3], 'Cam Z', -10.0, 0.0, valinit=-3.0)
    s_ty = Slider(slider_axes[4], 'Cam Y', -5.0, 5.0, valinit=0.5)
    s_tx = Slider(slider_axes[5], 'Cam X', -5.0, 5.0, valinit=0.5)

    s_crz = Slider(slider_axes[6], 'Cam Rot Z', -180, 180, valinit=0)
    s_cry = Slider(slider_axes[7], 'Cam Rot Y', -180, 180, valinit=0)
    s_crx = Slider(slider_axes[8], 'Cam Rot X', -180, 180, valinit=0)

    s_fy = Slider(slider_axes[9], 'Focal Y', 100, 500, valinit=250)
    s_fx = Slider(slider_axes[10], 'Focal X', 100, 500, valinit=250)
    
    # ***** Order must match unpacking in CubeProjection.__init__ *****
    sliders_tuple = (s_tx, s_ty, 
                     s_tz, s_rx, 
                     s_ry, s_rz, 
                     s_crx, s_cry, 
                     s_crz, s_fy, s_fx)

    # ***** Create CubeProjection instance *****
    cube_proj = CubeProjection(cube_vertices_world, 
                               cube_edges, 
                               ax, cx, cy, 
                               sliders_tuple)

    # ***** Bind slider updates to redraw function *****
    for s in sliders_tuple:
        s.on_changed(cube_proj.update_visualization)

    cube_proj.update_visualization(None) # Initial draw
    plt.show()
