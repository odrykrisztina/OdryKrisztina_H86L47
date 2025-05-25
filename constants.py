"""
Konstansok
"""

import numpy as np

# Hough transzformáció paraméterei
HOUGH_PARAMS = {
    'rho': 1,
    'theta': np.pi/180,
    'threshold': 80,
    'minLineLength': 150,
    'maxLineGap': 20
}

# Vonal detektálás paraméterei
MIN_LINE_LENGTH = 200
MIN_ANGLE_DIFF = 30
MAX_PARALLEL_DISTANCE = 50
MIN_LENGTH_RATIO = 0.5

# Könyvtár konstansok
INPUT_DIR = "./images"
OUTPUT_DIR = "./output"