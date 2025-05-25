# -*- coding: utf-8 -*-
"""
Pálcika detektáló program
"""

import cv2
from math import sqrt
import os
from line_detector import LineDetector
from image_processor import ImageProcessor
from constants import INPUT_DIR, OUTPUT_DIR, MIN_LINE_LENGTH


""" Kép mentése az output könyvtárba """
def save_image(image, base_filename, suffix):

    # Output könyvtár létrehozása, ha nem létezik
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Bemeneti fájlnév alapján kimeneti fájlnév generálása
    base_name = os.path.splitext(os.path.basename(base_filename))[0]
    output_filename = OUTPUT_DIR + f"/{base_name}_{suffix}.jpg"

    # Kép mentése
    cv2.imwrite(output_filename, image)


""" Főprogram """
def main():

    # Konzol menű
    print("\nVálasszon egy képet az elemzéshez:")
    print("1 - palcika1.jpg")
    print("2 - palcika2.jpg")
    print("3 - palcika3.jpg")
    print("4 - palcika4.jpg")

    # Felhasználói választás
    choice = input("Kérem adja meg a választott kép számát (1-4): ").strip()

    # A válasz elenőrzése
    if choice not in ['1', '2', '3', '4']:
        print("Érvénytelen választás! Kérem válasszon 1-4 között.")
        return

    # A kép nevének létrehozása, létezésének ellenőrzése
    filename = INPUT_DIR + f"/palcika{choice}.jpg"
    if not os.path.exists(filename):
        print(f"A kép nem található: {filename}")
        return

    # A kép betöltése, sikereségének ellenőrzése
    image = cv2.imread(filename)
    if image is None:
        print("Nem sikerult betolteni a kepet!")
        return

    # Kép előfeldolgozása
    print(f"Kép feldolgozása: {filename}")
    binary, edges = ImageProcessor.preprocess_image(image)

    # Köztes képek mentése
    save_image(binary, filename, "binary")
    save_image(edges, filename, "edges")

    # Vonalak detektálása
    lines = ImageProcessor.detect_lines(edges)

    # Vonalak összevonása
    merged_lines = LineDetector.merge_lines(lines)

    # Megvizsgáljuk hogy talált-e
    if merged_lines is None:
        print("Nem talaltam palcikakat!")
        return

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
            connected_lines = LineDetector.find_connected_lines(i, parallel_groups)
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
        angles.sort(key=lambda l: l[1])

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
            connected = LineDetector.find_connected_lines(idx, parallel_groups)
            group1_indices.update(connected)

        for idx in list(group2_indices):
            connected = LineDetector.find_connected_lines(idx, parallel_groups)
            group2_indices.update(connected)

        if group1_indices:
            line_groups.append([merged_lines[i] for i in group1_indices])
            used_lines.update(group1_indices)
        if group2_indices:
            line_groups.append([merged_lines[i] for i in group2_indices])
            used_lines.update(group2_indices)

    # Eredmény kiírása
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

    # Várakozás billentyűleütésre, majd az összes ablak becsukása
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()