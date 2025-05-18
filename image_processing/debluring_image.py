import cv2 as cv 
import numpy as np 

class ArtificialBlurring:
    """
    Perform artificial blurring and noise addition on an input image.

    This class supports Gaussian blur, directional box blur (horizontal/vertical/full),
    and additive Gaussian noise.
    """
    def __init__(self, img_path: str):
        """ 
        Initialize ArtificialBlurring
        
        Args:
            img_path (str): Path to the image to be processed.

        Raises:
            ValueError: If the image path is invalid or the image cannot be loaded.
        """
        self.img_path = img_path 
        self.img = cv.imread(img_path)
        if self.img is None:
            raise ValueError(f"Image at path '{img_path}' could not be loaded.")
        
    def gaussian_blur(self, ksize: int = 5, sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply 2D Gaussian blur using separable kernels.

        Args:
            ksize (int): Kernel size (must be odd). Default is 5.
            sigma (float): Standard deviation of the Gaussian kernel. Default is 1.0.

        Returns:
            blurred_image (np.ndarray): The blurred output image.
            gaussian_kernel (np.ndarray): The 2D Gaussian kernel used.
        """
        gk1d = cv.getGaussianKernel(ksize, sigma)
        # ***** Create 2D kernel by outer product *****
        gaussian_kernel =  gk1d @ gk1d.T
        # ***** Convolve the image with the kernel *****
        blurred_image = cv.filter2D(self.img, -1, gaussian_kernel)
        return blurred_image, gaussian_kernel
    
    def boxblur(self, ksize: int = 5, direction: str = 'horizontal') -> tuple[np.ndarray, np.ndarray]:
        """
        Apply directional or full box blur.

        Args:
            ksize (int): Kernel size. Default is 5.
            direction (str): 'horizontal', 'vertical', or any other string for full blur.

        Returns:
            blurred_image (np.ndarray): The blurred image.
            kernel (np.ndarray): The kernel used for the blur.
        """
        # ***** Create direction-specific box kernel *****
        if direction == 'horizontal': 
            kernel = np.ones((1, ksize), dtype=np.float32) / ksize
        elif direction == 'vertical':
            kernel = np.ones((ksize, 1), dtype=np.float32) / ksize
        else: 
            kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
            
        blurred_image = cv.filter2D(self.img, -1, kernel)
        return blurred_image, kernel
    
    def add_noise(self, image: np.ndarray, mean: float = 0, std: float = 10) -> np.ndarray:
        """
        Add Gaussian noise to an image.

        Args:
            image (np.ndarray): Input image to add noise to.
            mean (float): Mean of the Gaussian noise. Default is 0.
            std (float): Standard deviation of the Gaussian noise. Default is 10.

        Returns:
            noisy_image (np.ndarray): Image with added noise.
        """
        noise = np.zeros_like(image, dtype=np.float32)
        # ***** Generate Gaussian noise *****
        cv.randn(noise, mean, std)
        # ***** Add and clip noise *****
        noisy_image = cv.add(image.astype(np.float32), noise)
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image
    
class Deblurring:
    """
    Perform deblurring using Wiener deconvolution on a blurry image.
    """
    def __init__(self, image: np.ndarray):
        """ 
        Initialize Deblurring 
        
        Args:
            image (np.ndarray): Input blurry image.

        Raises:
            ValueError: If the image is None.
        """
        if image is None:
            raise ValueError("Input image is None.")
        self.blurry_image = image

    def pad_and_shift_kernel(self, kh: int, kw: int, shape: tuple[int, int], kernel: np.ndarray) -> np.ndarray:
        """
        Pad and shift the kernel to center it in frequency domain.

        Args:
            kh, kw (int): Kernel height and width.
            shape (tuple): Target shape for padding.
            kernel (np.ndarray): Original kernel.

        Returns:
            padded (np.ndarray): Zero-padded and shifted kernel.
        """
        padded = np.zeros(shape, dtype=np.float32)
        padded[:kh, :kw] = kernel
        # ***** Center the kernel using roll *****
        padded = np.roll(padded, -kh // 2, axis=0)
        padded = np.roll(padded, -kw // 2, axis=1)
        return padded
    
    def wiener_deconvolution(self, kernel: np.ndarray, K: float = 0.01) -> np.ndarray:
        """
        Apply Wiener deconvolution to a blurry image.

        Args:
            kernel (np.ndarray): Blur kernel used during the blurring step.
            K (float): Constant to prevent division by zero. Default is 0.01.

        Returns:
            restored_image (np.ndarray): Deblurred output image.
        """
        kernel = kernel / np.sum(kernel)
        kh, kw = kernel.shape 
        img = self.blurry_image 
        if len(img.shape) == 3 and img.shape[2] == 3: 
            restored_channels = []
            for c in range(3): 
                # ***** Fourier transform of image and kernel *****
                channel = img[:, :, c].astype(np.float32)
                G = np.fft.fft2(channel)
                H = np.fft.fft2(self.pad_and_shift_kernel(kh, kw, channel.shape, kernel))
                H_conj = np.conj(H)
                # ***** Construct Wiener filter *****
                W_filter = H_conj / (np.abs(H)**2 + K)
                F_hat = W_filter * G
                restored = np.fft.ifft2(F_hat)
                restored = np.abs(restored)
                restored = np.clip(restored, 0, 255).astype(np.uint8)
                restored_channels.append(restored)
            return cv.merge(restored_channels)
        
class BlurDeblurController:
    """
    GUI controller for applying and tuning blur and deblurring operations.

    This class handles GUI interaction using OpenCV trackbars and displays
    the original, blurred, and deblurred images side by side.

    Args:
        image_path (str): Path to the image.
    """

    def __init__(self, image_path: str):
        self.blur_type = 'gaussian'
        self.blurrer = ArtificialBlurring(image_path)
        self.original_img = self.blurrer.img
        self.kernel = None
        self.blurred_img = None
    
    def add_title_to_image(self, img: np.ndarray, title: str, position: tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        Draw a title text on the image.

        Args:
            img (np.ndarray): Input image.
            title (str): Title to overlay.
            position (tuple): Position for the title text.

        Returns:
            labeled_image (np.ndarray): Image with overlaid title.
        """
        return cv.putText(img.copy(), title, position, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def update(self, val=None) -> None:
        """
        Update image display based on current trackbar values.

        This function reads the trackbar values for blur kernel and Wiener constant,
        applies blur and deblurring accordingly, and updates the display window.
        """
        try:
            ksize = cv.getTrackbarPos('Kernel Size', 'Wiener Deconvolution') * 2 + 1
            sigma = cv.getTrackbarPos('Sigma', 'Wiener Deconvolution') / 10.0
            K = cv.getTrackbarPos('K (x1000)', 'Wiener Deconvolution') / 1000.0
        except cv.error as error:
            # ***** Graceful fail if GUI elements not yet ready *****
            print("Trackbars not ready yet:", error)
            return
         
        # ***** Apply selected blur type *****
        if self.blur_type == 'gaussian': 
            self.blurred_img, self.kernel = self.blurrer.gaussian_blur(ksize, sigma)
        elif self.blur_type == 'box':
            self.blurred_img, self.kernel = self.blurrer.boxblur(ksize)
            
        # ***** Apply Wiener deconvolution *****
        deblurring = Deblurring(self.blurred_img)
        deblurred_img = deblurring.wiener_deconvolution(self.kernel, K)
        
        # ***** Annotate and stack images *****
        original = self.add_title_to_image(self.original_img, 'Original')
        blurred = self.add_title_to_image(self.blurred_img, 'Blurred')
        deblurred = self.add_title_to_image(deblurred_img, 'Deblurred')

        stacked = np.hstack((original, blurred, deblurred))
        cv.imshow('Wiener Deconvolution', stacked)

    def set_blur_type(self, blur_type: str) -> None:
        """
        Set the type of blur to apply.

        Args:
            blur_type (str): Either 'gaussian' or 'box'.
        """
        self.blur_type = blur_type
        print(f"Switched to {blur_type} blur")
        self.update()

if __name__ == '__main__':
    # ***** Define image path and deblur controller instance *****
    image_path = 'images/image.jpg'
    controller = BlurDeblurController(image_path)

    cv.namedWindow('Wiener Deconvolution', cv.WINDOW_NORMAL)
    cv.resizeWindow('Wiener Deconvolution', 1200, 400)

    # ***** Trackbars use lambdas to call the controller's update method *****
    cv.createTrackbar('Kernel Size', 'Wiener Deconvolution', 2, 10, lambda val: controller.update(val))
    cv.createTrackbar('Sigma', 'Wiener Deconvolution', 10, 50, lambda val: controller.update(val))
    cv.createTrackbar('K (x1000)', 'Wiener Deconvolution', 10, 100, lambda val: controller.update(val))
    
    # ***** Initial processing update *****
    cv.waitKey(30)
    controller.update()
   
   # ***** Options to choose blur type *****
    print("Press 'g' for Gaussian, 'b' for Box blur. Press ESC to exit.")
    while True:
        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('g'):
            controller.set_blur_type('gaussian')
        elif key == ord('b'):
            controller.set_blur_type('box')

    cv.destroyAllWindows()
    
   
    
   
    
   