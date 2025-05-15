import cv2 as cv 
import numpy as np
import time

class ColorObjectTracker: 
    """
    Designed primarily to detect a red water bottle in real-time via webcam feed. It utilizes HSV
    thresholding with trackbars for calibration, applies noise reduction, and identifies circular
    contours to localize and highlight the object on screen.

    """
    def __init__(self, cap: cv.VideoCapture) -> None:
        """ 
        Initialize the ColorObjectTracker object with video capture
        
        Args:
            cap: cv.VideoCapture instance representing the camera feed.
        """
        self.cap = cap
    
    def nothing(self, x: int) -> None:
        """
        Dummy callback function used as a placeholder for trackbar creation.

        Args:
            x: Trackbar position value.
        """
        pass
    
    def trackbars(self) -> None:
        """
        Creates an OpenCV window with HSV threshold trackbars to interactively
        adjust the color filtering range.
        """
        cv.namedWindow("Trackbars")
        cv.resizeWindow("Trackbars", 400, 300)

        # ***** Create HSV trackbars to allow interactive tuning *****
        cv.createTrackbar("LH", "Trackbars", 0, 179, self.nothing)
        cv.createTrackbar("LS", "Trackbars", 0, 255, self.nothing)
        cv.createTrackbar("LV", "Trackbars", 0, 255, self.nothing)
        cv.createTrackbar("UH", "Trackbars", 179, 179, self.nothing)
        cv.createTrackbar("US", "Trackbars", 255, 255, self.nothing)
        cv.createTrackbar("UV", "Trackbars", 255, 255, self.nothing)

    def remove_noise(self, mask: np.ndarray) -> np.ndarray:
        """
        Applies morphological operations to remove noise from the binary mask.

        Args:
            mask: Binary mask obtained from color segmentation.

        Returns:
            mask: Cleaned binary mask after erosion and dilation.
        """
        # ***** Use erosion and dilation to clean up small noise *****
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.erode(mask, kernel, iterations=1)
        mask = cv.dilate(mask, kernel, iterations=2)
        return mask
    
    def get_color_tracking(self) -> None:
        """
        Main loop that performs real-time object detection and tracking.

        Continuously captures frames from the camera, converts them to HSV color space,
        applies red color segmentation (handling both hue ranges of red),
        filters noise, detects contours, and highlights circular objects (e.g., a red bottle cap).
        Displays FPS and result windows until the ESC key is pressed.
        """
        self.trackbars()

        while True:
            start_time = time.time()
            ret, frame = self.cap.read()

            # ***** Exit loop if frame is not read correctly *****
            if not ret:
                break

            # ***** Convert to HSV color space for better color segmentation *****
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            # ***** Get HSV bounds from trackbars *****
            # lh = cv.getTrackbarPos("LH", "Trackbars")
            ls = cv.getTrackbarPos("LS", "Trackbars")
            lv = cv.getTrackbarPos("LV", "Trackbars")
            # uh = cv.getTrackbarPos("UH", "Trackbars")
            us = cv.getTrackbarPos("US", "Trackbars")
            uv = cv.getTrackbarPos("UV", "Trackbars")

            # ***** Define two ranges to cover red hue wrap-around *****
            lower_red1 = np.array([0, ls, lv])
            upper_red1 = np.array([10, us, uv])
            lower_red2 = np.array([160, ls, lv])
            upper_red2 = np.array([179, us, uv])

            # ***** Create masks for both red hue ranges *****
            mask1 = cv.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv.inRange(hsv, lower_red2, upper_red2)
            mask = cv.bitwise_or(mask1, mask2)

            # ***** Remove noise from mask before contour detection *****
            mask = self.remove_noise(mask)

            # ***** Find external contours in the cleaned binary mask *****
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # ***** Shape detection *****
                area = cv.contourArea(contour)
                x, y, w, h = cv.boundingRect(contour)
                aspect_ratio = float(w) / h
                if area > 1000 and 0.2 < aspect_ratio < 0.5:
                    cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                    cv.putText(frame, "Bottle", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # ***** Calculate and display Frames Per Second (FPS) *****
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # ***** Display the original frame and the binary mask *****
            cv.imshow("Original", frame)
            cv.imshow("Mask", mask)

            key = cv.waitKey(1)
            if key == 27:  # ESC key
                break

        # ***** Release resources and close windows *****
        self.cap.release()
        cv.destroyAllWindows()
                                

if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    color_track = ColorObjectTracker(cap)
    color_track.get_color_tracking()
