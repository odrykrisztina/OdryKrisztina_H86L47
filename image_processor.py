"""
ImageProcessor osztály a képfeldolgozási műveletek végrehajtásához
"""

import cv2
import numpy as np
from constants import HOUGH_PARAMS


class ImageProcessor:
    """Képfeldolgozásért felelős osztály"""

    @staticmethod
    def preprocess_image(image):
        """Kép előfeldolgozása"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        edges = cv2.Canny(binary, 30, 150)
        return binary, edges

    @staticmethod
    def detect_lines(edges):
        """Vonalak detektálása Hough transzformációval"""
        return cv2.HoughLinesP(edges, **HOUGH_PARAMS)

    @staticmethod
    def generate_distinct_colors(n):
        """Egyedi színek generálása"""
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            saturation = 255
            value = 255

            hsv_color = np.uint8([[[hue, saturation, value]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, bgr_color)))

        return colors