"""
LineDetector osztály
------------------
A vonalak matematikai feldolgozásáért és geometriai számításokért felelős osztály.
Statikus metódusokat tartalmaz a vonalak hosszának, szögének számításához,
kereszteződések detektálásához és párhuzamos vonalak kezeléséhez.
"""

import numpy as np
from math import sqrt
from constants import MIN_ANGLE_DIFF


class LineDetector:

    """ Vonal hosszának számítása euklideszi távolság alapján """
    @staticmethod
    def line_length(line):
        """
        Args:
            line: Vonal koordinátái [[x1, y1, x2, y2]] formátumban

        Returns:
            float: A vonal hossza pixelekben
        """
        x1, y1, x2, y2 = line[0]
        return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


    """ Vonal szögének számítása a vízszintes tengelyhez képest """
    @staticmethod
    def get_line_angle(line):
        """
        Args:
            line: Vonal koordinátái [[x1, y1, x2, y2]] formátumban

        Returns:
            float: A vonal szöge fokban (0-180 között)
        """
        x1, y1, x2, y2 = line[0]
        # Arkusz tangens számítása, konvertálás fokokba
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        # Negatív szögek korrigálása a 0-180 tartományba
        if angle < 0:
            angle += 180
        return angle


    """ Két vonal metszéspontjának meghatározása """
    @staticmethod
    def find_intersection(line1, line2):
        """
        Args:
            line1: Első vonal koordinátái [[x1, y1, x2, y2]] formátumban
            line2: Második vonal koordinátái [[x1, y1, x2, y2]] formátumban

        Returns:
            tuple vagy None: (x, y) metszéspont koordinátái, vagy None ha nincs metszéspont
        """
        x1, y1, x2, y2 = line1[0]
        x3, y3, x4, y4 = line2[0]

        # Szögek ellenőrzése
        angle1 = LineDetector.get_line_angle(line1)
        angle2 = LineDetector.get_line_angle(line2)
        angle_diff = abs(angle1 - angle2)

        # 90 foknál nagyobb szögkülönbség esetén a kiegészítő szöget vesszük
        if angle_diff > 90:
            angle_diff = 180 - angle_diff

        # Ha a szögkülönbség kisebb mint a minimum, nincs érvényes kereszteződés
        if angle_diff < MIN_ANGLE_DIFF:
            return None

        # Metszéspont számítása lineáris egyenletrendszer megoldásával
        # A vonalak egyenletei: ax + by = c formában
        a1 = y2 - y1
        b1 = x1 - x2
        c1 = a1 * x1 + b1 * y1
        a2 = y4 - y3
        b2 = x3 - x4
        c2 = a2 * x3 + b2 * y3
        det = a1 * b2 - a2 * b1

        # Ha a determináns 0, a vonalak párhuzamosak
        if det == 0:
            return None

        # Metszéspont koordinátáinak számítása
        x = (b2 * c1 - b1 * c2) / det
        y = (a1 * c2 - a2 * c1) / det

        # Ellenőrizzük, hogy a metszéspont a vonalszakaszokon van-e
        if (min(x1, x2) <= x <= max(x1, x2) and
                min(y1, y2) <= y <= max(y1, y2) and
                min(x3, x4) <= x <= max(x3, x4) and
                min(y3, y4) <= y <= max(y3, y4)):
            return int(x), int(y)

        return None


    """ Ellenőrzi, hogy két vonal párhuzamos és közel van-e egymáshoz """
    @staticmethod
    def are_lines_parallel_and_close(line1, line2, max_angle_diff=15, max_distance=60):
        """
        Args:
            line1: Első vonal koordinátái [[x1, y1, x2, y2]] formátumban
            line2: Második vonal koordinátái [[x1, y1, x2, y2]] formátumban
            max_angle_diff: Maximális megengedett szögeltérés fokban (alapértelmezett: 15)
            max_distance: Maximális megengedett távolság pixelben (alapértelmezett: 60)

        Returns:
            bool: True ha a vonalak párhuzamosak és közel vannak, False egyébként
        """

        # Vonalak szögeinek összehasonlítása
        angle1 = LineDetector.get_line_angle(line1)
        angle2 = LineDetector.get_line_angle(line2)

        # Szögkülönbség kiszámítása
        angle_diff = abs(angle1 - angle2)

        # 90 foknál nagyobb szögkülönbség esetén a kiegészítő szöget vesszük
        if angle_diff > 90:
            angle_diff = 180 - angle_diff

        # Ha a szögeltérés nagyobb mint a megengedett, nem párhuzamosak
        if angle_diff > max_angle_diff:
            return False

        x1, y1, x2, y2 = line1[0]
        x3, y3, x4, y4 = line2[0]

        # Vonalak középpontjainak kiszámítása
        mid1_x = (x1 + x2) / 2
        mid1_y = (y1 + y2) / 2
        mid2_x = (x3 + x4) / 2
        mid2_y = (y3 + y4) / 2

        # Középpontok távolságának ellenőrzése
        mid_dist = sqrt((mid1_x - mid2_x) ** 2 + (mid1_y - mid2_y) ** 2)

        # Vonalak irányvektorainak számítása
        dir1_x = x2 - x1
        dir1_y = y2 - y1
        dir2_x = x4 - x3
        dir2_y = y4 - y3

        # Irányvektorok normalizálása
        len1 = sqrt(dir1_x ** 2 + dir1_y ** 2)
        len2 = sqrt(dir2_x ** 2 + dir2_y ** 2)
        dir1_x, dir1_y = dir1_x / len1, dir1_y / len1
        dir2_x, dir2_y = dir2_x / len2, dir2_y / len2

        # Skaláris szorzat számítása az irányok összehasonlításához
        dot_product = dir1_x * dir2_x + dir1_y * dir2_y

        # Ha a skaláris szorzat negatív, megfordítjuk az előjelét
        if dot_product < 0:
            dot_product = -dot_product

        # Vonalak közötti minimális távolság számítása
        dist1 = abs((x3 - x1) * dir1_y - (y3 - y1) * dir1_x)
        dist2 = abs((x4 - x1) * dir1_y - (y4 - y1) * dir1_x)
        min_dist = min(dist1, dist2)

        # Minden feltétel ellenőrzése
        return (min_dist < max_distance and
                mid_dist < max_distance * 2 and
                dot_product > 0.85)


    """ Vonalak összevonása párhuzamosság és közelség alapján """
    @staticmethod
    def merge_lines(lines):
        """
        Args:
            lines: Vonalak listája, minden vonal [[x1, y1, x2, y2]] formátumban

        Returns:
            list: Összevont vonalak listája [[x1, y1, x2, y2]] formátumban
        """

        # Ha nem vonal visszatér
        if lines is None:
            return None

        # Összegző
        merged_lines = []
        used = [False] * len(lines)

        # Vonalak rendezése hossz szerint csökkenő sorrendbe
        line_lengths = [(i, LineDetector.line_length(line)) for i, line in enumerate(lines)]
        line_lengths.sort(key=lambda x: x[1], reverse=True)

        # Végigmegyünk minden vonalon
        for i, _ in line_lengths:

            # Ha viszgálva van, ugrik a következő vonalra
            if used[i]:
                continue

            # Új csoport kezdése az aktuális vonallal
            current_group = [lines[i][0]]
            used[i] = True
            base_line = lines[i]

            # Hasonló vonalak keresése
            for j, _ in line_lengths:

                # Ha viszgálva van, ugrik a következő vonalra
                if used[j]:
                    continue

                # Ha párhuzamos és közel van, hozzáadjuk a csoporthoz
                if LineDetector.are_lines_parallel_and_close(base_line, lines[j]):
                    current_group.append(lines[j][0])
                    used[j] = True

            # Ha találtunk vonalakat a csoportban
            if current_group:

                # Végpontok összegyűjtése
                x_coords = []
                y_coords = []
                for line in current_group:
                    x_coords.extend([line[0], line[2]])
                    y_coords.extend([line[1], line[3]])

                # Legtávolabbi pontok keresése
                distances = []
                points = list(zip(x_coords, y_coords))
                for p1 in points:
                    for p2 in points:
                        if p1 != p2:
                            dist = sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                            distances.append((dist, p1, p2))

                # A legtávolabbi pontokat összekötő vonal létrehozása
                if distances:
                    max_dist, p1, p2 = max(distances, key=lambda x: x[0])
                    merged_lines.append(np.array([[int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])]]))

        return merged_lines


    """ Összefüggő (párhuzamos) vonalak keresése """
    @staticmethod
    def find_connected_lines(start_idx, parallel_groups):
        """
        Args:
            start_idx: Kezdő vonal indexe
            parallel_groups: Párhuzamos vonalak csoportjai (szótár: index -> párhuzamos vonalak indexei)

        Returns:
            set: Az összefüggő vonalak indexeinek halmaza
        """

        # Összefüggő vonalak halmaza
        connected = set()

        # Feldolgozásra váró vonalak
        to_process = {start_idx}

        # Amíg van feldolgozandó vonal
        while to_process:
            current_idx = to_process.pop()
            if current_idx in connected:
                continue

            # Aktuális vonal hozzáadása a kapcsolódó vonalakhoz
            connected.add(current_idx)

            # Párhuzamos vonalak hozzáadása a feldolgozandó vonalakhoz
            for parallel_idx in parallel_groups.get(current_idx, set()):
                if parallel_idx not in connected:
                    to_process.add(parallel_idx)

        return connected