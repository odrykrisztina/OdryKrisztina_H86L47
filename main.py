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
    """
    Args:
        image: A lementendő kép
        base_filename:  Bemeneti fájl neve
        suffix: A fájlnévhez hozáadandó toldalék
    """
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

    """ Vonalpárok vizsgálata """
    # Végigmegyünk az összes vonalon
    for i in range(len(merged_lines)):

        # Inicializáljuk az i-edik vonalhoz tartozó adatstruktúrákat
        intersection_map[i] = set()     # Az i-edik vonallal kereszteződő vonalak indexei
        parallel_groups[i] = set()      # Az i-edik vonallal párhuzamos vonalak indexei
        intersection_points[i] = []     # Az i-edik vonal kereszteződési pontjai

        # Csak az i-nél nagyobb indexű vonalakkal hasonlítjuk össze
        # Ha már tudjuk, hogy az i-edik és j-edik vonal kereszteződik/párhuzamos,
        # akkor nem kell újra megvizsgálni fordítva.
        for j in range(i + 1, len(merged_lines)):

            # Keressük a kereszteződést az i-edik és j-edik vonal között
            intersection = LineDetector.find_intersection(merged_lines[i], merged_lines[j])

            # Ha van kereszteződés
            if intersection:

                # Eltároljuk a kereszteződési pontot
                intersections.append(intersection)

                # Az i-edik vonalhoz hozzáadjuk a j-edik vonalat mint kereszteződőt
                intersection_map[i].add(j)

                # Az i-edik vonalhoz eltároljuk a kereszteződési pontot
                intersection_points[i].append(intersection)

                # Ha a j-edik vonal még nem szerepel a kereszteződési térképben
                if j not in intersection_map:
                    # Létrehozzuk
                    intersection_map[j] = set()
                    intersection_points[j] = []

                # A j-edik vonalhoz eltároljuk az i-edik vonalat
                intersection_map[j].add(i)

                # A j-edik vonalhoz eltároljuk a kereszteződési pontot
                intersection_points[j].append(intersection)

            # Ha nincs kereszteződés, de párhuzamosak és közel vannak
            elif LineDetector.are_lines_parallel_and_close(merged_lines[i], merged_lines[j]):

                # Az i-edik vonalhoz hozzáadjuk a j-edik vonalat mint párhuzamost
                parallel_groups[i].add(j)

                # Ha a j-edik vonal még nem szerepel a párhuzamos csoportokban létrehozzuk
                if j not in parallel_groups:
                    parallel_groups[j] = set()

                # A j-edik vonalhoz is eltároljuk az i-edik vonalat mint párhuzamost
                parallel_groups[j].add(i)

    """ Vonalak csoportosítása """

    # Összegző a vonalcsoport tárolására
    line_groups = []

    # A már feldolgozott vonalak indexeinek halmaza
    used_lines = set()

    """ Nem kereszteződő vonalak csoportosítása """
    # Végigmegyünk az összes vonalon
    for i in range(len(merged_lines)):

        # Ha a vonal még nem volt feldolgozva és nincs kereszteződése
        if i not in used_lines and not intersection_map[i]:

            # Megkeressük az összes vele párhuzamos és összefüggő vonalat
            connected_lines = LineDetector.find_connected_lines(i, parallel_groups)

            # Létrehozunk egy új vonalcsoportot a kapcsolódó vonalakból
            line_group = [merged_lines[idx] for idx in connected_lines]

            # Hozzáadjuk az új csoportot a csoportok listájához
            line_groups.append(line_group)

            # Megjelöljük az összes kapcsolódó vonalat feldolgozottként
            used_lines.update(connected_lines)

    """ Kereszteződő vonalak kezelése """
    # A kereszteződési pontokhoz tartozó vonalak tárolása
    # Kulcs: kereszteződési pont koordinátái (x,y)
    # Érték: az itt kereszteződő vonalak indexeinek halmaza
    crossing_groups = {}

    # Végigmegyünk az összes vonalon
    for i in range(len(merged_lines)):

        # Ha ez a vonal még nem volt feldolgozva és van kereszteződési pontja
        if i not in used_lines and intersection_points[i]:

            # Végigmegyünk a vonal összes kereszteződési pontján
            for point in intersection_points[i]:

                # A pont koordinátáit használjuk kulcsként
                point_key = (point[0], point[1])

                # Ha ez egy új kereszteződési pont, inicializáljuk a halmazt
                if point_key not in crossing_groups:
                    crossing_groups[point_key] = set()

                # Hozzáadjuk a vonalat a kereszteződési ponthoz tartozó halmazhoz
                crossing_groups[point_key].add(i)

    """ Közeli kereszteződések összevonása """
    merged_crossing_groups = []         # Itt tároljuk az összevont kereszteződési csoportokat
    used_points = set()                 # A már feldolgozott kereszteződési pontok

    # Végigmegyünk az összes kereszteződési ponton és a hozzájuk tartozó vonalakon
    for point, lines in crossing_groups.items():

        # Ha ezt a pontot még nem dolgoztuk fel
        if point not in used_points:

            # Új csoport létrehozása az aktuális pont vonalaival
            current_group = set(lines)
            used_points.add(point)

            # Keressük a közeli kereszteződési pontokat
            for other_point, other_lines in crossing_groups.items():

                # Ha a másik pont még nem volt feldolgozva
                if other_point not in used_points:

                    # Kiszámoljuk a két pont közötti távolságot
                    dist = sqrt((point[0] - other_point[0]) ** 2 + (point[1] - other_point[1]) ** 2)

                    # Ha a pontok elég közel vannak egymáshoz (50 pixel)
                    if dist < 50:

                        # A közeli pont vonalait hozzáadjuk az aktuális csoporthoz
                        current_group.update(other_lines)
                        used_points.add(other_point)

            # Az összevont csoportot hozzáadjuk a listához
            merged_crossing_groups.append(current_group)

    """ Kereszteződő vonalak csoportosítása szög alapján """

    # Végigmegyünk az összevont kereszteződési csoportokon
    for group in merged_crossing_groups:

        # Kiszámoljuk minden vonalhoz a szögét és rendezzük őket
        angles = [(i, LineDetector.get_line_angle(merged_lines[i])) for i in group]
        angles.sort(key=lambda l: l[1])

        # Első csoport inicializálása az első vonallal
        group1_indices = set()
        base_angle = angles[0][1]
        group1_indices.add(angles[0][0])

        # A többi vonal vizsgálata
        for idx, angle in angles[1:]:

            # Szögkülönbség számítása
            angle_diff = abs(angle - base_angle)

            # 90 foknál nagyobb szögkülönbség esetén a kiegészítő szöget vesszük
            if angle_diff > 90:
                angle_diff = 180 - angle_diff

            # Ha a szögkülönbség kisebb mint 25 fok, akkor ugyanabba a csoportba tartozik
            if angle_diff < 25:
                group1_indices.add(idx)

        # A maradék vonalak a második csoportba kerülnek
        group2_indices = group - group1_indices

        # Az első csoport vonalaihoz tartozó párhuzamos vonalak hozzáadása
        for idx in list(group1_indices):
            connected = LineDetector.find_connected_lines(idx, parallel_groups)
            group1_indices.update(connected)

        # A második csoport vonalaihoz tartozó párhuzamos vonalak hozzáadása
        for idx in list(group2_indices):
            connected = LineDetector.find_connected_lines(idx, parallel_groups)
            group2_indices.update(connected)

        # Ha vannak vonalak az első csoportban, hozzáadjuk a vonalcsoportokhoz
        if group1_indices:
            line_groups.append([merged_lines[i] for i in group1_indices])
            used_lines.update(group1_indices)

        # Ha vannak vonalak a második csoportban, hozzáadjuk a vonalcsoportokhoz
        if group2_indices:
            line_groups.append([merged_lines[i] for i in group2_indices])
            used_lines.update(group2_indices)

    # Eredmény kiírása
    print(f"Talalt palcikak szama: {len(line_groups)}")
    print(f"Talalt keresztezodesek szama: {len(intersections)}")

    """ Eredmények megjelenítése """
    # Az eredeti kép másolata, amin megjelenítjük az eredményeket
    result = image.copy()

    # Egyedi színek generálása minden vonalcsoporthoz
    # Minimum 9 szín kell, hogy elég különböző szín legyen
    colors = ImageProcessor.generate_distinct_colors(max(len(line_groups), 9))

    # Vonalcsoportok megjelenítése különböző színekkel
    for i, group in enumerate(line_groups):
        # Az aktuális csoport színe
        color = colors[i]

        # A csoport összes vonalának megrajzolása
        for line in group:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), color, 2)

        # Ha van vonal a csoportban, kiírjuk a vonalak számát
        if group:
            # A csoport középpontjának kiszámítása
            center_x = center_y = 0
            for line in group:
                x1, y1, x2, y2 = line[0]
                center_x += (x1 + x2) / 2
                center_y += (y1 + y2) / 2
            # Átlagolás
            center_x /= len(group)
            center_y /= len(group)

            # Vonalak számának kiírása a csoport középpontjába
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