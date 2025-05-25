"""
Konstansok
"""

import numpy as np

# Hough transzformáció paraméterei
HOUGH_PARAMS = {
    'rho': 1,               # A Hough tér felbontása pixelekben
    'theta': np.pi/180,     # A Hough tér szögfelbontása radiánban
    'threshold': 80,        # Minimális metszéspont szám a Hough térben
    'minLineLength': 150,   # Minimális vonalhossz
    'maxLineGap': 20        # Maximális rés két vonalszegmens között
}

# Vonal detektálás paraméterei
MIN_LINE_LENGTH = 200           # Minimális elfogadható vonalhossz pixelben
MIN_ANGLE_DIFF = 30             # Minimális szögkülönbség kereszteződéseknél (fok)
MAX_PARALLEL_DISTANCE = 50      # Maximális távolság párhuzamos vonalak között (pixel)
MIN_LENGTH_RATIO = 0.5          # Minimális hosszarány a vonalak összehasonlításánál

# Könyvtár konstansok
INPUT_DIR = "./images"
OUTPUT_DIR = "./output"