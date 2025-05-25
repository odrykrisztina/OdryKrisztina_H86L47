# -*- coding: utf-8 -*-
"""
Pálcika detektáló program
------------------------
A program képes detektálni és megszámolni a képen található pálcikákat,
kezelve a kereszteződéseket és párhuzamos vonalakat.

Funkciók:
- Felhasználói választás az elemzendő képről (palcika1.jpg vagy palcika2.jpg)
- Automatikus pálcika detektálás Canny él-detektálással és Hough transzformációval
- Kereszteződések felismerése és kezelése
- Párhuzamos vonalak csoportosítása
- Interaktív eredmény megjelenítés átméretezhető ablakban
- Eredmények automatikus mentése az output mappába

Használat:
1. Indítsa el a programot
2. Válassza ki az elemezni kívánt képet (1 vagy 2)
3. Az eredmény ablakban megtekintheti a detektált pálcikákat
4. Az ablak átméretezhető és a címsorban mozgatható
5. Bármely billentyű lenyomásával bezárhatja az ablakot
"""

import cv2
import numpy as np
from math import sqrt
import os

# Konstansok
HOUGH_PARAMS = {
    'rho': 1,
    'theta': np.pi / 180,
    'threshold': 80,
    'minLineLength': 150,
    'maxLineGap': 20
}

MIN_LINE_LENGTH = 200
MIN_ANGLE_DIFF = 30
MAX_PARALLEL_DISTANCE = 50
MIN_LENGTH_RATIO = 0.5

# Könyvtár konstansok
INPUT_DIR = "./images"
OUTPUT_DIR = "./output"


class LineDetector:
    """Vonalak detektálásáért és feldolgozásáért felelős osztály"""

    @staticmethod
    def line_length(line):
        """Vonal hosszának számítása"""
        x1, y1, x2, y2 = line[0]
        return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @staticmethod
    def get_line_angle(line):
        """Vonal szögének számítása"""
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if angle < 0:
            angle += 180
        return angle

    @staticmethod
    def find_intersection(line1, line2):
        """Két vonal metszéspontjának meghatározása"""
        x1, y1, x2, y2 = line1[0]
        x3, y3, x4, y4 = line2[0]

        # Szögek ellenőrzése
        angle1 = LineDetector.get_line_angle(line1)
        angle2 = LineDetector.get_line_angle(line2)
        angle_diff = abs(angle1 - angle2)
        if angle_diff > 90:
            angle_diff = 180 - angle_diff

        if angle_diff < MIN_ANGLE_DIFF:
            return None

        # Metszéspont számítása
        A1 = y2 - y1
        B1 = x1 - x2
        C1 = A1 * x1 + B1 * y1
        A2 = y4 - y3
        B2 = x3 - x4
        C2 = A2 * x3 + B2 * y3
        det = A1 * B2 - A2 * B1

        if det == 0:
            return None

        x = (B2 * C1 - B1 * C2) / det
        y = (A1 * C2 - A2 * C1) / det

        if (min(x1, x2) <= x <= max(x1, x2) and
                min(y1, y2) <= y <= max(y1, y2) and
                min(x3, x4) <= x <= max(x3, x4) and
                min(y3, y4) <= y <= max(y3, y4)):
            return (int(x), int(y))

        return None

    @staticmethod
    def are_lines_parallel_and_close(line1, line2, max_angle_diff=15, max_distance=60):
        """Ellenőrzi, hogy két vonal párhuzamos és közel van-e egymáshoz"""
        angle1 = LineDetector.get_line_angle(line1)
        angle2 = LineDetector.get_line_angle(line2)

        angle_diff = abs(angle1 - angle2)
        if angle_diff > 90:
            angle_diff = 180 - angle_diff

        if angle_diff > max_angle_diff:
            return False

        x1, y1, x2, y2 = line1[0]
        x3, y3, x4, y4 = line2[0]

        # Vonalak középpontjai
        mid1_x = (x1 + x2) / 2
        mid1_y = (y1 + y2) / 2
        mid2_x = (x3 + x4) / 2
        mid2_y = (y3 + y4) / 2

        # Középpontok távolsága
        mid_dist = sqrt((mid1_x - mid2_x) ** 2 + (mid1_y - mid2_y) ** 2)

        # Vonalak irányvektora
        dir1_x = x2 - x1
        dir1_y = y2 - y1
        dir2_x = x4 - x3
        dir2_y = y4 - y3

        # Irányvektor normalizálása
        len1 = sqrt(dir1_x ** 2 + dir1_y ** 2)
        len2 = sqrt(dir2_x ** 2 + dir2_y ** 2)
        dir1_x, dir1_y = dir1_x / len1, dir1_y / len1
        dir2_x, dir2_y = dir2_x / len2, dir2_y / len2

        # Skaláris szorzat az irány ellenőrzéséhez
        dot_product = dir1_x * dir2_x + dir1_y * dir2_y

        if dot_product < 0:
            dir2_x, dir2_y = -dir2_x, -dir2_y
            dot_product = -dot_product

        # Vonalak közötti minimális távolság számítása
        dist1 = abs((x3 - x1) * dir1_y - (y3 - y1) * dir1_x)
        dist2 = abs((x4 - x1) * dir1_y - (y4 - y1) * dir1_x)
        min_dist = min(dist1, dist2)

        return (min_dist < max_distance and
                mid_dist < max_distance * 2 and
                dot_product > 0.85)

    @staticmethod
    def merge_lines(lines, min_distance=60, min_angle_diff=15):
        """Vonalak összevonása"""
        if lines is None:
            return None

        merged_lines = []
        used = [False] * len(lines)

        # Vonalak rendezése hossz szerint
        line_lengths = [(i, LineDetector.line_length(line)) for i, line in enumerate(lines)]
        line_lengths.sort(key=lambda x: x[1], reverse=True)

        for i, _ in line_lengths:
            if used[i]:
                continue

            current_group = [lines[i][0]]
            used[i] = True
            base_line = lines[i]

            # Hasonló vonalak keresése
            for j, _ in line_lengths:
                if used[j]:
                    continue

                if LineDetector.are_lines_parallel_and_close(base_line, lines[j]):
                    current_group.append(lines[j][0])
                    used[j] = True

            if current_group:
                x_coords = []
                y_coords = []
                for line in current_group:
                    x_coords.extend([line[0], line[2]])
                    y_coords.extend([line[1], line[3]])

                # Legtávolabbi pontok összekötése
                distances = []
                points = list(zip(x_coords, y_coords))
                for p1 in points:
                    for p2 in points:
                        if p1 != p2:
                            dist = sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                            distances.append((dist, p1, p2))

                if distances:
                    max_dist, p1, p2 = max(distances, key=lambda x: x[0])
                    merged_lines.append(np.array([[int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])]]))

        return merged_lines

    @staticmethod
    def find_connected_lines(start_idx, merged_lines, intersection_map, parallel_groups):
        """Összefüggő vonalak keresése"""
        connected = set()
        to_process = {start_idx}

        while to_process:
            current_idx = to_process.pop()
            if current_idx in connected:
                continue

            connected.add(current_idx)

            for parallel_idx in parallel_groups.get(current_idx, set()):
                if parallel_idx not in connected:
                    to_process.add(parallel_idx)

        return connected


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


def save_image(image, base_filename, suffix):
    """Kép mentése az output könyvtárba"""
    # Output könyvtár létrehozása, ha nem létezik
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Bemeneti fájlnév alapján kimeneti fájlnév generálása
    base_name = os.path.splitext(os.path.basename(base_filename))[0]
    output_filename = os.path.join(OUTPUT_DIR, f"{base_name}_{suffix}.jpg")

    # Kép mentése
    cv2.imwrite(output_filename, image)
    print(f"Kep mentve: {output_filename}")


def main():
    """Főprogram"""
    # Felhasználói választás
    print("\nVálasszon egy képet az elemzéshez:")
    print("1 - palcika1.jpg")
    print("2 - palcika2.jpg")
    print("3 - palcika3.jpg")
    print("4 - palcika4.jpg")

    choice = input("Kérem adja meg a választott kép számát (1-4): ").strip()

    if choice not in ['1', '2', '3', '4']:
        print("Érvénytelen választás! Kérem válasszon 1 vagy 2 között.")
        return

    # Kép betöltése
    filename = os.path.join(INPUT_DIR, f"palcika{choice}.jpg")
    if not os.path.exists(filename):
        print(f"A kép nem található: {filename}")
        return

    image = cv2.imread(filename)
    if image is None:
        print("Nem sikerult betolteni a kepet!")
        return

    # Kép előfeldolgozása
    binary, edges = ImageProcessor.preprocess_image(image)

    # Köztes képek mentése
    save_image(binary, filename, "binary")
    save_image(edges, filename, "edges")

    # Vonalak detektálása
    lines = ImageProcessor.detect_lines(edges)

    # Vonalak összevonása
    merged_lines = LineDetector.merge_lines(lines, min_distance=50, min_angle_diff=25)

    if merged_lines is not None:
        # Vonalak szűrése hossz alapján
        merged_lines = [line for line in merged_lines if LineDetector.line_length(line) > MIN_LINE_LENGTH]

        # Kereszteződések és párhuzamos vonalak keresése
        intersections = []
        intersection_points = {}
        intersection_map = {}
        parallel_groups = {}

        # Vonalpárok vizsgálata
        for i in range(len(merged_lines)):
            intersection_map[i] = set()
            parallel_groups[i] = set()
            intersection_points[i] = []

            for j in range(i + 1, len(merged_lines)):
                intersection = LineDetector.find_intersection(merged_lines[i], merged_lines[j])
                if intersection:
                    intersections.append(intersection)
                    intersection_map[i].add(j)
                    intersection_points[i].append(intersection)
                    if j not in intersection_map:
                        intersection_map[j] = set()
                        intersection_points[j] = []
                    intersection_map[j].add(i)
                    intersection_points[j].append(intersection)
                elif LineDetector.are_lines_parallel_and_close(merged_lines[i], merged_lines[j]):
                    parallel_groups[i].add(j)
                    if j not in parallel_groups:
                        parallel_groups[j] = set()
                    parallel_groups[j].add(i)

        # Vonalak csoportosítása
        line_groups = []
        used_lines = set()

        # Nem kereszteződő vonalak csoportosítása
        for i in range(len(merged_lines)):
            if i not in used_lines and not intersection_map[i]:
                connected_lines = LineDetector.find_connected_lines(i, merged_lines, intersection_map, parallel_groups)
                line_group = [merged_lines[idx] for idx in connected_lines]
                line_groups.append(line_group)
                used_lines.update(connected_lines)

        # Kereszteződő vonalak kezelése
        crossing_groups = {}

        for i in range(len(merged_lines)):
            if i not in used_lines and intersection_points[i]:
                for point in intersection_points[i]:
                    point_key = (point[0], point[1])
                    if point_key not in crossing_groups:
                        crossing_groups[point_key] = set()
                    crossing_groups[point_key].add(i)

        # Közeli kereszteződések összevonása
        merged_crossing_groups = []
        used_points = set()

        for point, lines in crossing_groups.items():
            if point not in used_points:
                current_group = set(lines)
                used_points.add(point)

                for other_point, other_lines in crossing_groups.items():
                    if other_point not in used_points:
                        dist = sqrt((point[0] - other_point[0]) ** 2 + (point[1] - other_point[1]) ** 2)
                        if dist < 50:
                            current_group.update(other_lines)
                            used_points.add(other_point)

                merged_crossing_groups.append(current_group)

        # Kereszteződő vonalak csoportosítása
        for group in merged_crossing_groups:
            angles = [(i, LineDetector.get_line_angle(merged_lines[i])) for i in group]
            angles.sort(key=lambda x: x[1])

            group1_indices = set()
            base_angle = angles[0][1]
            group1_indices.add(angles[0][0])

            for idx, angle in angles[1:]:
                angle_diff = abs(angle - base_angle)
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff
                if angle_diff < 25:
                    group1_indices.add(idx)

            group2_indices = group - group1_indices

            for idx in list(group1_indices):
                connected = LineDetector.find_connected_lines(idx, merged_lines, intersection_map, parallel_groups)
                group1_indices.update(connected)

            for idx in list(group2_indices):
                connected = LineDetector.find_connected_lines(idx, merged_lines, intersection_map, parallel_groups)
                group2_indices.update(connected)

            if group1_indices:
                line_groups.append([merged_lines[i] for i in group1_indices])
                used_lines.update(group1_indices)
            if group2_indices:
                line_groups.append([merged_lines[i] for i in group2_indices])
                used_lines.update(group2_indices)

        print(f"Talalt palcikak szama: {len(line_groups)}")
        print(f"Talalt keresztezodesek szama: {len(intersections)}")

        # Eredmény megjelenítése
        result = image.copy()
        colors = ImageProcessor.generate_distinct_colors(max(len(line_groups), 9))

        for i, group in enumerate(line_groups):
            color = colors[i]
            for line in group:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), color, 2)

            if group:
                center_x = center_y = 0
                for line in group:
                    x1, y1, x2, y2 = line[0]
                    center_x += (x1 + x2) / 2
                    center_y += (y1 + y2) / 2
                center_x /= len(group)
                center_y /= len(group)

                cv2.putText(result, f"Vonalak: {len(group)}",
                            (int(center_x), int(center_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Kereszteződések rajzolása
        for x, y in intersections:
            cv2.circle(result, (x, y), 5, (0, 0, 255), -1)

        # Eredmény kép mentése
        save_image(result, filename, "result")

        # Ablak létrehozása és fókuszba helyezése
        window_name = "Detektalt palcikak"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, result)

        # Ablak előtérbe hozása
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        # Várakozás billentyűleütésre
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Nem talaltam palcikakat!")


if __name__ == "__main__":
    main()