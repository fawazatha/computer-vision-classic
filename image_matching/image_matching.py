import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 

class ImageMatching: 
    """ 
    Perform keypoint-based image matching using ORB and RANSAC filtering.
    Ratio test filtering, and homography-based inlier selection with RANSAC.
    """
    def __init__(self, image_a: str, image_b: str):
        """  
        Initialize ImageMatching object 
        
        Args:
            image_a (str): Path to the first image (usually the object to detect).
            image_b (str): Path to the second image (usually the scene).
        """
        self.image_a = cv.imread(image_a)
        self.image_a = cv.cvtColor(self.image_a, cv.COLOR_BGR2GRAY)
        self.image_b = cv.imread(image_b)
        self.image_b = cv.cvtColor(self.image_b, cv.COLOR_BGR2GRAY)
        print("Image A shape:", self.image_a.shape)
        print("Image B shape:", self.image_b.shape)

    def keypoint_detection(self) -> None:
        """
        Detect ORB keypoints and descriptors for both images.

        Returns:
            None
        """
        # ***** Initialize ORB and detect keypoints + descriptors *****
        orb = cv.ORB_create()
        self.keypoints_a, self.descriptor_a = orb.detectAndCompute(self.image_a, None)
        self.keypoints_b, self.descriptor_b = orb.detectAndCompute(self.image_b, None)

    def match_with_ratio_test(self, ratio_thresh: float = 0.75) -> None:
        """
        Perform brute-force KNN matching followed by Lowe's ratio test.

        Args:
            ratio_thresh (float): Ratio threshold for filtering matches.

        Returns:
            None
        """
        # ***** Match descriptors using BFMatcher with Hamming norm *****
        bruteforce = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        knn_matches = bruteforce.knnMatch(self.descriptor_a, self.descriptor_b, k=2)

        # ***** Apply Lowe's ratio test to select good matches *****
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        self.ratio_test_matches = good_matches
    
    def ransac_filtering(self) -> None:
        """
        Apply RANSAC to filter out false matches using homography estimation.

        Returns:
            None
        """
        # ***** Extract matched keypoints for homography *****
        pts1 = np.float32([self.keypoints_a[m.queryIdx].pt for m in self.ratio_test_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([self.keypoints_b[m.trainIdx].pt for m in self.ratio_test_matches]).reshape(-1, 1, 2)

        # ***** Estimate homography and filter inliers *****
        H, mask = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)
        inliers = [m for m, keep in zip(self.ratio_test_matches, mask.ravel()) if keep]
        self.inlier_matches = inliers
        
    def draw_matches(self, title: str = 'Matches', max_matches: int = 50, dot_size: int = 4) -> None:
        """
        Draw matches between keypoints of two images using matplotlib.

        Args:
            title (str): Title of the plotted figure.
            max_matches (int): Maximum number of inlier matches to draw.
            dot_size (int): Size of keypoint dots in the visualization.

        Returns:
            None
        """
        # ***** Convert grayscale images back to RGB for display *****
        img1 = cv.cvtColor(self.image_a, cv.COLOR_BGR2RGB)
        img2 = cv.cvtColor(self.image_b, cv.COLOR_BGR2RGB)

        # ***** Create a canvas to display both images side by side *****
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        height = max(h1, h2)
        canvas = np.zeros((height, w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img1
        canvas[:h2, w1:] = img2

        # ***** Plot matches using matplotlib *****
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(canvas)
        ax.axis('off')

        for match in self.inlier_matches[:max_matches]:
            idx1 = match.queryIdx
            idx2 = match.trainIdx

            pt1 = tuple(np.round(self.keypoints_a[idx1].pt).astype(int))
            pt2 = tuple(np.round(self.keypoints_b[idx2].pt).astype(int))
            pt2_shifted = (pt2[0] + w1, pt2[1])  # shift x-coordinates for image B

            ax.plot([pt1[0], pt2_shifted[0]], [pt1[1], pt2_shifted[1]], 'r-', linewidth=0.6)
            ax.scatter(*pt1, color='lime', s=dot_size)
            ax.scatter(*pt2_shifted, color='cyan', s=dot_size)

        plt.title(title)
        plt.tight_layout()
        plt.show()
    
if __name__ == '__main__':
    image_path_a = 'images/image_a.png'
    image_path_b = 'images/image_b.png'
    
    matcher = ImageMatching(image_path_a, image_path_b)
    matcher.keypoint_detection()
    matcher.match_with_ratio_test()
    matcher.ransac_filtering()
    matcher.draw_matches()
