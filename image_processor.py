"""
ImageProcessor osztály
--------------------
A képfeldolgozási műveletek végrehajtásáért felelős osztály.
Statikus metódusokat tartalmaz a képek előfeldolgozásához, vonalak
detektálásához és a megjelenítéshez szükséges színek generálásához.
"""

import cv2
import numpy as np
from constants import HOUGH_PARAMS


class ImageProcessor:

    """ Képfeldolgozási műveletek végrehajtása """
    @staticmethod
    def preprocess_image(image):
        """
        Args:
            image: BGR színtérben lévő bemeneti kép

        Returns:
            tuple: (binary, edges)
                - binary: Binarizált kép
                - edges: Éldetektált kép
        """

        # Szürkeárnyalatos konverzió
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Gauss-féle elmosás a zaj csökkentésére
        # 5x5-ös kernel méret, 0 szigma érték (automatikus számítás)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptív küszöbölés a binarizáláshoz
        # - 255: maximális érték
        # - ADAPTIVE_THRESH_GAUSSIAN_C: Gaussian-alapú adaptív küszöbölés
        # - THRESH_BINARY_INV: Invertált bináris kép
        # - 11: A környezet mérete (11x11 pixel)
        # - 2: Konstans kivonás a számított küszöbértékből
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Morfológiai nyitás a zaj további csökkentésére
        # 3x3-as kernel az apró zajok eltávolításához
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Canny él-detektálás
        # - 30: alsó küszöbérték
        # - 150: felső küszöbérték
        edges = cv2.Canny(binary, 30, 150)

        return binary, edges


    """ Vonalak detektálása Hough transzformációval """
    @staticmethod
    def detect_lines(edges):
        """
        Args:
            edges: Éldetektált bináris kép

        Returns:
            numpy.ndarray: Detektált vonalak listája, minden vonal [x1, y1, x2, y2] formátumban
            vagy None, ha nem talált vonalakat
        """
        # A HoughLinesP függvény paraméterei a constants.py HOUGH_PARAMS szótárából:
        # - rho: A Hough tér felbontása pixelekben
        # - theta: A Hough tér szögfelbontása radiánban
        # - threshold: Minimális metszéspont szám a Hough térben
        # - minLineLength: Minimális vonalhossz
        # - maxLineGap: Maximális rés két vonalszegmens között
        return cv2.HoughLinesP(edges, **HOUGH_PARAMS)


    """ Egyedi színek generálása """
    @staticmethod
    def generate_distinct_colors(n):
        """
        Args:
            n: A generálandó színek száma

        Returns:
            list: BGR színek listája, minden szín (B, G, R) tuple-ként
        """
        colors = []
        for i in range(n):
            # HSV színtérben generáljuk a színeket az egyenletes eloszlás érdekében
            # - Hue: 0-180 között egyenletesen elosztva
            # - Saturation: Maximális (255)
            # - Value: Maximális (255)
            hue = int(180 * i / n)
            saturation = 255
            value = 255

            # Konvertálás HSV színtérből BGR színtérbe
            hsv_color = np.uint8([[[hue, saturation, value]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]

            # Az értékek egész számmá konvertálása és tárolása tuple-ként
            colors.append(tuple(map(int, bgr_color)))

        return colors