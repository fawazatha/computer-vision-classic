import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 

class GrabCut: 
    """ 
    Interactive image segmentation using the GrabCut algorithm.
        
    Allows users to select an initial ROI, apply GrabCut, refine segmentation using foreground/background scribbles,
    and visualize intermediate and final segmentation results.
    """
    def __init__(self, 
                 image_path: str, 
                 brush_size: int = 5):
        """
        Initialize GrabCut object with image and parameters. 
        This sets up the image, mask, and models needed for GrabCut segmentation.    

        Args:
            image_path (str): Path to the input image.
            brush_size (int): Radius of the brush used for drawing foreground/background scribbles.

        Attributes:
            image (np.ndarray): Original image loaded from path.
            clone (np.ndarray): Clone used for drawing annotations.
            mask (np.ndarray): GrabCut mask.
            bgdModel, fgdModel (np.ndarray): Background/foreground models for GrabCut.
            brush_size (int): Radius of brush for drawing.
            value (int): Current drawing mode (GC_FGD or GC_BGD).
        """
        self.image = cv.imread(image_path) 
        self.clone = self.image.copy()
        self.drawing = False
        self.value = cv.GC_FGD  # Start with foreground scribble
        self.brush_size = brush_size
        self.scribbles_done = False
        self.bgdModel = np.zeros((1, 65), np.float64)
        self.fgdModel = np.zeros((1, 65), np.float64)
         
    def draw_rectangle(self): 
        """
        Allow user to select a rectangular ROI on the image for initial segmentation.
        """
        # ***** Let user select ROI *****
        self.roi = cv.selectROI('Select ROI', self.image, fromCenter=False, showCrosshair=True) 
        cv.destroyWindow('Select ROI') 
        
        # ***** Extract coordinates and initialize mask within ROI *****
        self.x, self.y, self.w, self.h = self.roi
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.mask[self.y:self.y + self.h, self.x:self.x + self.w] = cv.GC_PR_FGD
        
    def apply_grabcut(self, mode: int = cv.GC_INIT_WITH_RECT, itercount: int = 5) -> None:
        """
        Apply GrabCut algorithm to the image using the current mask or ROI.

        Args:
            mode (int): GrabCut mode (cv.GC_INIT_WITH_RECT or cv.GC_INIT_WITH_MASK).
            itercount (int): Number of iterations for GrabCut refinement.
        
        Returns: 
            None
        """
        # ***** Run GrabCut based on selected mode *****
        cv.grabCut(self.image, self.mask, None if mode == cv.GC_INIT_WITH_MASK else self.roi, 
                   self.bgdModel, self.fgdModel, itercount, mode)
        
        if mode == cv.GC_INIT_WITH_RECT: 
            # ***** Save initial mask and visualization before user scribbles *****
            self.initial_mask_binary = np.where((self.mask == cv.GC_BGD) | (self.mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
            self.initial_foreground = self.image * self.initial_mask_binary[:, :, np.newaxis]
        
        # ***** Final binary mask and segmented foreground image *****
        self.final_mask_binary = np.where((self.mask==cv.GC_PR_BGD)|(self.mask==cv.GC_BGD), 0, 1).astype('uint8')
        self.segmented_foreground = self.image * self.final_mask_binary[:, :, np.newaxis]
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Callback for drawing foreground/background scribbles using the mouse.
        """
        if event == cv.EVENT_LBUTTONDOWN:
            # ***** Start drawing circle (scribble) on mask and clone image *****
            self.drawing = True 
            cv.circle(self.mask, (x, y), self.brush_size, self.value, -1)
            cv.circle(self.clone, (x, y), self.brush_size, (0, 255, 0) if self.value == cv.GC_FGD else (0, 0, 255), -1)
            
        elif event == cv.EVENT_MOUSEMOVE and self.drawing: 
            # ***** Continue drawing while mouse moves *****
            cv.circle(self.mask, (x, y), self.brush_size, self.value, -1)
            cv.circle(self.clone, (x, y), self.brush_size, (0, 255, 0) if self.value == cv.GC_FGD else (0, 0, 255), -1)
            
        elif event == cv.EVENT_LBUTTONUP:
            # ***** Stop drawing when mouse released *****
            self.drawing = False 
            
    def get_scribbles(self) -> None: 
        """
        Allow user to draw foreground (f) and background (b) scribbles for refinement.
        Press 'q' to finish.
        """
        print("Please draw scribbles on the image. 'f' for FG, 'b' for BG, 'q' to finish.")
        cv.namedWindow('Scribble Input')
        cv.setMouseCallback('Scribble Input', self.mouse_callback)
        
        # ***** Wait for user input (f, b, or q) *****
        while True: 
            cv.imshow('Scribble Input', self.clone)
            key = cv.waitKey(1) & 0xFF
            
            if key == ord('f'):
                self.value = cv.GC_FGD
            elif key == ord('b'):
                self.value = cv.GC_BGD
            elif key == ord('q'):
                break
        
    def visualize(self) -> None: 
        """
        Display original image, ROI, initial segmentation, and final result.
        """
        fig, axs = plt.subplots(1, 4, figsize=(18, 6))
        
        # ***** Show original image *****
        axs[0].imshow(cv.cvtColor(self.image, cv.COLOR_BGR2RGB))
        axs[0].set_title('Original Image')
        axs[0].axis('off')
        
        # ***** Show image with ROI rectangle *****
        image_with_roi = self.image.copy()
        cv.rectangle(image_with_roi, (self.x, self.y), (self.x+self.w, self.y+self.h), (0, 255, 0), 2)
        axs[1].imshow(cv.cvtColor(image_with_roi, cv.COLOR_BGR2RGB))
        axs[1].set_title('Original with ROI')
        axs[1].axis('off')
        
        # ***** Show segmentation before scribbles *****
        axs[2].imshow(cv.cvtColor(self.initial_foreground, cv.COLOR_BGR2RGB))
        axs[2].set_title('Initial GrabCut (Before Brush)')
        axs[2].axis('off')
    
        # ***** Show final result after refinement *****
        axs[3].imshow(cv.cvtColor(self.segmented_foreground, cv.COLOR_BGR2RGB))
        axs[3].set_title('Final GrabCut (After Brush)')
        axs[3].axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # ***** Define image path and run full GrabCut workflow *****
    img_path = 'images/image.jpg'
    grabcut = GrabCut(img_path)
    grabcut.draw_rectangle()                        # Select ROI
    grabcut.apply_grabcut()                         # Initial GrabCut segmentation
    grabcut.get_scribbles()                         # Manual foreground/background annotation
    grabcut.apply_grabcut(mode=cv.GC_INIT_WITH_MASK)  # Refine with mask
    grabcut.visualize()                             # Display results