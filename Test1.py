import cv2
import numpy as np
import os


class VideoAnalyzer:

    def __init__(self, video_path):

        # ==========================================================
        # VIDEO
        # ==========================================================
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            print("ERROR: Could not open video!")
            print("Path:", os.path.abspath(video_path))
            print("File exists:", os.path.exists(video_path))
            return

        print("Video opened successfully!")

        self.current_frame_idx = 0
        self.is_paused = False
        self.current_frame = None
        self.current_roi = None
        self.current_roi_origin = (0, 0)

        # ==========================================================
        # ROI
        # ==========================================================
        self.roi = None
        self.is_roi_selected = False

        # ==========================================================
        # FROZEN ROI
        # ==========================================================
        self.frozen_roi = None
        self.frozen_zoom = 4.0

        # ==========================================================
        # STABILIZATION
        # ==========================================================
        self.prev_gray = None
        self.camera_x = 0.0
        self.camera_y = 0.0

        self.smoothing = 0.92
        self.dead_zone = 0.15
        self.max_shift = 15.0

        # ==========================================================
        # DETECTION MODE
        # B = Brightness
        # C = Color
        # A = Brightness + Color
        # ==========================================================
        self.detection_mode = "B"

        # ==========================================================
        # COLORS - HSV
        # ==========================================================
        self.color_ranges = {
            "BLUE": (
                np.array([100, 60, 35], dtype=np.uint8),
                np.array([140, 255, 255], dtype=np.uint8)
            ),

            "GREEN": (
                np.array([35, 55, 35], dtype=np.uint8),
                np.array([88, 255, 255], dtype=np.uint8)
            ),

            "TURQUOISE": (
                np.array([80, 45, 35], dtype=np.uint8),
                np.array([105, 255, 255], dtype=np.uint8)
            ),

            "YELLOW": (
                np.array([15, 55, 35], dtype=np.uint8),
                np.array([40, 255, 255], dtype=np.uint8)
            ),

            "RED1": (
                np.array([0, 55, 35], dtype=np.uint8),
                np.array([12, 255, 255], dtype=np.uint8)
            ),

            "RED2": (
                np.array([168, 55, 35], dtype=np.uint8),
                np.array([179, 255, 255], dtype=np.uint8)
            )
        }

        # ==========================================================
        # BRIGHTNESS / PIXEL DETECTION
        # ==========================================================
        self.brightness_factor = 1.12
        self.local_brightness_offset = 18
        self.min_area = 2
        self.max_area = 100000

        # Small colored/bright pixels are intentionally preserved.
        self.morph_kernel = np.ones((2, 2), np.uint8)

        # ==========================================================
        # ARRAYS
        # ==========================================================
        self.detected_objects = []
        self.manual_points = []

        # Deleted automatic detections are stored by CENTER,
        # so a slightly changing bounding box will still remain deleted.
        self.deleted_objects = []

        # ==========================================================
        # MANUAL MODES
        # ==========================================================
        self.manual_add_mode = False
        self.manual_delete_mode = False

        # ==========================================================
        # TEMPORAL FILTER
        # ==========================================================
        # Kept at 1 so small/moving pixels are not lost.
        self.previous_detections = []
        self.required_frames = 1

        # ==========================================================
        # ZOOM
        # ==========================================================
        self.zoom = 4.0
        self.main_zoom = 1.0

        # ==========================================================
        # LETTER MAPPING PANEL
        # ==========================================================
        self.selected_letter = "A"
        self.letter_panel_width = 420
        self.letter_canvas_size = 360
        self.letter_font = cv2.FONT_HERSHEY_SIMPLEX
        self.letter_font_scale = 10.0
        self.letter_thickness = 22

    # ==========================================================
    # BRIGHTNESS DETECTION
    # ==========================================================
    def detect_bright_pixels(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Global brightness
        mean_brightness = float(np.mean(gray))
        global_threshold = min(
            mean_brightness * self.brightness_factor,
            250
        )

        _, global_mask = cv2.threshold(
            gray,
            global_threshold,
            255,
            cv2.THRESH_BINARY
        )

        # Local brightness:
        # detects a pixel that is brighter than its nearby environment,
        # which is much more useful inside a selected ROI.
        local_mean = cv2.GaussianBlur(
            gray,
            (0, 0),
            7
        )

        local_mask = (
            gray.astype(np.int16)
            >
            (local_mean.astype(np.int16) + self.local_brightness_offset)
        ).astype(np.uint8) * 255

        # Use both tests, but do not require both.
        mask = cv2.bitwise_or(
            global_mask,
            local_mask
        )

        return mask

    # ==========================================================
    # COLOR DETECTION
    # ==========================================================
    def detect_color(self, image):

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        final_mask = np.zeros(
            hsv.shape[:2],
            dtype=np.uint8
        )

        # All requested colors
        for name in (
            "BLUE",
            "GREEN",
            "TURQUOISE",
            "YELLOW"
        ):
            lower, upper = self.color_ranges[name]

            mask = cv2.inRange(
                hsv,
                lower,
                upper
            )

            final_mask = cv2.bitwise_or(
                final_mask,
                mask
            )

        # Red wraps around HSV 0/179.
        lower, upper = self.color_ranges["RED1"]
        red1 = cv2.inRange(hsv, lower, upper)

        lower, upper = self.color_ranges["RED2"]
        red2 = cv2.inRange(hsv, lower, upper)

        red_mask = cv2.bitwise_or(
            red1,
            red2
        )

        final_mask = cv2.bitwise_or(
            final_mask,
            red_mask
        )

        return final_mask

    # ==========================================================
    # OBJECT / PIXEL DETECTION
    # ==========================================================
    def detect_objects(self, image):

        if image is None or image.size == 0:
            return []

        bright_mask = self.detect_bright_pixels(image)
        color_mask = self.detect_color(image)

        if self.detection_mode == "B":
            mask = bright_mask

        elif self.detection_mode == "C":
            mask = color_mask

        elif self.detection_mode == "A":
            # For A, keep colored pixels OR locally bright pixels.
            # This avoids losing valid colored pixels that are not
            # globally bright enough.
            mask = cv2.bitwise_or(
                bright_mask,
                color_mask
            )

        else:
            mask = bright_mask

        # ======================================================
        # NOISE REMOVAL WITHOUT DESTROYING SMALL PIXELS
        # ======================================================
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.morph_kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            self.morph_kernel
        )

        # Slight dilation connects neighboring pixels only when
        # they really belong to the same small region.
        connect_kernel = np.ones(
            (2, 2),
            np.uint8
        )

        mask = cv2.dilate(
            mask,
            connect_kernel,
            iterations=1
        )

        # ======================================================
        # FIND CONNECTED COMPONENTS
        # ======================================================
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )

        objects = []

        for label in range(1, num_labels):

            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])

            if area < self.min_area:
                continue

            if area > self.max_area:
                continue

            objects.append(
                [x, y, w, h]
            )

        return objects

    # ==========================================================
    # COLOR OF DETECTED PIXEL
    # ==========================================================
    def get_pixel_color(
        self,
        image,
        x,
        y,
        w,
        h
    ):

        if image is None or image.size == 0:
            return "UNKNOWN"

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        cx = int(x + w / 2)
        cy = int(y + h / 2)

        cx = max(
            0,
            min(
                cx,
                hsv.shape[1] - 1
            )
        )

        cy = max(
            0,
            min(
                cy,
                hsv.shape[0] - 1
            )
        )

        pixel = hsv[cy, cx]

        hue = int(pixel[0])
        saturation = int(pixel[1])

        if saturation < 45:
            return "UNKNOWN"

        if hue <= 12 or hue >= 168:
            return "RED"

        if 15 <= hue <= 40:
            return "YELLOW"

        if 35 <= hue <= 88:
            # Separate turquoise from green.
            if 80 <= hue <= 105:
                return "TURQUOISE"
            return "GREEN"

        if 100 <= hue <= 140:
            return "BLUE"

        return "UNKNOWN"

    # ==========================================================
    # TEMPORAL FILTER
    # ==========================================================
    def filter_detections(self, objects):

        if self.required_frames <= 1:
            self.previous_detections = objects.copy()
            return objects

        if not objects:
            self.previous_detections = []
            return []

        confirmed = []

        for obj in objects:

            x, y, w, h = obj

            cx = x + w / 2
            cy = y + h / 2

            found_count = 1

            for old in self.previous_detections:

                ox, oy, ow, oh = old

                ocx = ox + ow / 2
                ocy = oy + oh / 2

                distance = np.sqrt(
                    (cx - ocx) ** 2 +
                    (cy - ocy) ** 2
                )

                if distance < 15:
                    found_count += 1
                    break

            if found_count >= self.required_frames:
                confirmed.append(obj)

        self.previous_detections = objects.copy()

        return confirmed

    # ==========================================================
    # CHECK IF OBJECT WAS MANUALLY DELETED
    # ==========================================================
    def is_deleted_object(self, obj):

        x, y, w, h = obj

        cx = x + w / 2
        cy = y + h / 2

        for dx, dy in self.deleted_objects:

            distance = np.sqrt(
                (cx - dx) ** 2 +
                (cy - dy) ** 2
            )

            # The tolerance is deliberately larger than the
            # detection box size so the deletion survives small
            # frame-to-frame bbox changes.
            if distance < max(
                12,
                min(35, max(w, h) * 2)
            ):
                return True

        return False

    # ==========================================================
    # CAMERA STABILIZATION
    # ==========================================================
    def stabilize_frame(self, frame):

        try:

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            if self.prev_gray is None:

                self.prev_gray = gray.copy()
                self.camera_x = 0.0
                self.camera_y = 0.0

                return frame

            previous_points = cv2.goodFeaturesToTrack(
                self.prev_gray,
                maxCorners=300,
                qualityLevel=0.01,
                minDistance=12,
                blockSize=7
            )

            if previous_points is None:

                self.prev_gray = gray.copy()
                return frame

            current_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                previous_points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS |
                    cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01
                )
            )

            if (
                current_points is None
                or status is None
            ):
                self.prev_gray = gray.copy()
                return frame

            valid = (
                status.ravel() == 1
            )

            good_previous = previous_points[valid]
            good_current = current_points[valid]

            if len(good_previous) < 10:

                self.prev_gray = gray.copy()
                return frame

            movement = (
                good_current -
                good_previous
            )

            dx_values = movement[:, 0]
            dy_values = movement[:, 1]

            dx_median = np.median(
                dx_values
            )

            dy_median = np.median(
                dy_values
            )

            # Remove outliers
            distance = np.sqrt(
                (dx_values - dx_median) ** 2 +
                (dy_values - dy_median) ** 2
            )

            good = (
                distance < 3.0
            )

            if np.sum(good) >= 8:

                dx = np.median(
                    dx_values[good]
                )

                dy = np.median(
                    dy_values[good]
                )

            else:

                dx = dx_median
                dy = dy_median

            dx = float(
                np.clip(
                    dx,
                    -self.max_shift,
                    self.max_shift
                )
            )

            dy = float(
                np.clip(
                    dy,
                    -self.max_shift,
                    self.max_shift
                )
            )

            if abs(dx) < self.dead_zone:
                dx = 0.0

            if abs(dy) < self.dead_zone:
                dy = 0.0

            self.camera_x = (
                self.smoothing * self.camera_x
                +
                (1 - self.smoothing) * dx
            )

            self.camera_y = (
                self.smoothing * self.camera_y
                +
                (1 - self.smoothing) * dy
            )

            matrix = np.float32([
                [1, 0, -self.camera_x],
                [0, 1, -self.camera_y]
            ])

            stabilized = cv2.warpAffine(
                frame,
                matrix,
                (
                    frame.shape[1],
                    frame.shape[0]
                ),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )

            self.prev_gray = gray.copy()

            return stabilized

        except Exception as error:

            print(
                "Stabilization error:",
                error
            )

            return frame

    # ==========================================================
    # CONVERT MOUSE POSITION TO ORIGINAL VIDEO POSITION
    # ==========================================================
    def mouse_to_video_point(self, x, y):

        # Normal view
        if (
            self.main_zoom <= 1
            or self.roi is None
        ):
            return int(x), int(y)

        rx, ry, rw, rh = self.roi

        zoom_w = int(rw * self.main_zoom)
        zoom_h = int(rh * self.main_zoom)

        # The zoomed ROI is displayed at the top-left corner.
        if (
            0 <= x < zoom_w
            and 0 <= y < zoom_h
        ):

            local_x = int(
                x / self.main_zoom
            )

            local_y = int(
                y / self.main_zoom
            )

            return (
                int(rx + local_x),
                int(ry + local_y)
            )

        return int(x), int(y)

    # ==========================================================
    # MOUSE
    # ==========================================================
    def mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        param
    ):

        if not self.is_roi_selected:
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        x, y = self.mouse_to_video_point(
            x,
            y
        )

        # Keep manual clicks inside the ROI.
        rx, ry, rw, rh = self.roi

        if not (
            rx <= x < rx + rw
            and
            ry <= y < ry + rh
        ):
            return

        # ======================================================
        # MANUAL ADD - M
        # ======================================================
        if self.manual_add_mode:

            point = [
                int(x),
                int(y)
            ]

            already_exists = False

            for px, py in self.manual_points:

                distance = (
                    (px - x) ** 2 +
                    (py - y) ** 2
                )

                if distance < 25:
                    already_exists = True
                    break

            if not already_exists:

                self.manual_points.append(
                    point
                )

                print()
                print("MANUAL PIXEL ADDED")
                print("X =", x)
                print("Y =", y)
                print(
                    "Manual Array =",
                    self.manual_points
                )
                print()

        # ======================================================
        # MANUAL DELETE - X
        # ======================================================
        elif self.manual_delete_mode:

            # --------------------------------------------------
            # First delete a manually added point
            # --------------------------------------------------
            closest_index = -1
            closest_distance = float("inf")

            for i, point in enumerate(
                self.manual_points
            ):

                px, py = point

                distance = (
                    (px - x) ** 2 +
                    (py - y) ** 2
                )

                if distance < closest_distance:

                    closest_distance = distance
                    closest_index = i

            if (
                closest_index != -1
                and closest_distance < 400
            ):

                removed = self.manual_points.pop(
                    closest_index
                )

                print()
                print("MANUAL PIXEL DELETED")
                print("Removed =", removed)
                print(
                    "Manual Array =",
                    self.manual_points
                )
                print()

                return

            # --------------------------------------------------
            # Delete automatic detection by CENTER
            # --------------------------------------------------
            closest_object = None
            closest_distance = float("inf")

            for obj in self.detected_objects:

                ox, oy, ow, oh = obj

                center_x = ox + ow / 2
                center_y = oy + oh / 2

                distance = (
                    (center_x - x) ** 2 +
                    (center_y - y) ** 2
                )

                if distance < closest_distance:

                    closest_distance = distance
                    closest_object = obj

            if (
                closest_object is not None
                and closest_distance < 900
            ):

                ox, oy, ow, oh = closest_object

                center_x = ox + ow / 2
                center_y = oy + oh / 2

                center = (
                    float(center_x),
                    float(center_y)
                )

                # Avoid duplicates
                already_deleted = False

                for dx, dy in self.deleted_objects:

                    if np.sqrt(
                        (dx - center_x) ** 2 +
                        (dy - center_y) ** 2
                    ) < 10:

                        already_deleted = True
                        break

                if not already_deleted:

                    self.deleted_objects.append(
                        center
                    )

                if closest_object in self.detected_objects:

                    self.detected_objects.remove(
                        closest_object
                    )

                print()
                print("AUTOMATIC PIXEL DELETED")
                print("Removed =", closest_object)
                print(
                    "Deleted Centers =",
                    self.deleted_objects
                )
                print()

    # ==========================================================
    # DRAW MANUAL POINT
    # ==========================================================
    def draw_manual_point(
        self,
        image,
        px,
        py,
        offset_x=0,
        offset_y=0,
        scale=1.0
    ):

        local_x = int(
            (px - offset_x) * scale
        )

        local_y = int(
            (py - offset_y) * scale
        )

        size = max(
            3,
            int(5 * scale)
        )

        cv2.rectangle(
            image,
            (
                local_x - size,
                local_y - size
            ),
            (
                local_x + size,
                local_y + size
            ),
            (0, 0, 255),
            2
        )

    # ==========================================================
    # PRINT COORDINATES - P
    # ==========================================================
    def print_coordinates(self):

        print()
        print("======================================")
        print("          CURRENT COORDINATES")
        print("======================================")
        print(
            "Frame:",
            self.current_frame_idx
        )
        print(
            "Detection mode:",
            self.detection_mode
        )

        print()
        print("Automatic detections:")

        if not self.detected_objects:

            print("No automatic detection.")

        else:

            for i, obj in enumerate(
                self.detected_objects
            ):

                x, y, w, h = obj

                color = self.get_pixel_color(
                    self.current_roi,
                    x - self.current_roi_origin[0],
                    y - self.current_roi_origin[1],
                    w,
                    h
                )

                print(
                    i + 1,
                    "->",
                    "Color =", color,
                    "| X =", x,
                    "| Y =", y,
                    "| W =", w,
                    "| H =", h
                )

        print()
        print("Manual points:")

        if not self.manual_points:

            print("No manual points.")

        else:

            for i, point in enumerate(
                self.manual_points
            ):

                print(
                    i + 1,
                    "-> X =",
                    point[0],
                    "Y =",
                    point[1]
                )

        print()
        print("Deleted automatic centers:")

        if not self.deleted_objects:

            print("No deleted objects.")

        else:

            for i, center in enumerate(
                self.deleted_objects
            ):

                print(
                    i + 1,
                    "->",
                    center
                )

        print("======================================")
        print()

    # ==========================================================
    # DRAW ROI CONTENT
    # ==========================================================
    def build_zoom_roi(
        self,
        roi,
        x1,
        y1
    ):

        zoom_roi = roi.copy()

        # Automatic detections
        for gx, gy, gw, gh in (
            self.detected_objects
        ):

            local_x = int(
                gx - x1
            )

            local_y = int(
                gy - y1
            )

            cv2.rectangle(
                zoom_roi,
                (
                    local_x,
                    local_y
                ),
                (
                    local_x + gw,
                    local_y + gh
                ),
                (0, 255, 0),
                2
            )

            detected_color = self.get_pixel_color(
                roi,
                local_x,
                local_y,
                gw,
                gh
            )

            cv2.putText(
                zoom_roi,
                detected_color,
                (
                    local_x,
                    max(
                        15,
                        local_y - 5
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        # Manual points
        for px, py in self.manual_points:

            self.draw_manual_point(
                zoom_roi,
                px,
                py,
                x1,
                y1,
                1.0
            )

        zoomed_roi = cv2.resize(
            zoom_roi,
            None,
            fx=self.zoom,
            fy=self.zoom,
            interpolation=cv2.INTER_NEAREST
        )

        return zoomed_roi

    # ==========================================================
    # LETTER MAPPING PANEL
    # ==========================================================
    def build_letter_panel(self, source_image):
        """Resize the video/ROI onto a 2-D canvas and clip it by the typed glyph."""
        panel_w = self.letter_panel_width
        panel_h = self.letter_canvas_size + 110
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (22, 22, 22)

        cv2.putText(panel, "LETTER VIEW", (18, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(panel, f"Letter: {self.selected_letter}", (18, 57),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "Type A-Z / a-z", (18, 82), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (180, 180, 180), 1, cv2.LINE_AA)

        left, top = 30, 100
        right = min(left + self.letter_canvas_size, panel_w - 18)
        bottom = min(top + self.letter_canvas_size, panel_h - 18)
        cw, ch = right - left, bottom - top

        axis_color = (150, 150, 150)
        cv2.rectangle(panel, (left, top), (right, bottom), (70, 70, 70), 1)
        cv2.line(panel, (left, bottom), (right, bottom), axis_color, 1)
        cv2.line(panel, (left, top), (left, bottom), axis_color, 1)
        cv2.putText(panel, "X", (right - 14, bottom - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, axis_color, 1, cv2.LINE_AA)
        cv2.putText(panel, "Y", (left + 7, top + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, axis_color, 1, cv2.LINE_AA)

        glyph = np.zeros((ch, cw), dtype=np.uint8)
        scale = self.letter_font_scale
        (tw, th), baseline = cv2.getTextSize(self.selected_letter, self.letter_font,
                                               scale, self.letter_thickness)
        while (tw > cw - 30 or th + baseline > ch - 30) and scale > 1.0:
            scale *= 0.90
            (tw, th), baseline = cv2.getTextSize(self.selected_letter, self.letter_font,
                                                   scale, self.letter_thickness)
        tx = max(5, (cw - tw) // 2)
        ty = max(th + 5, (ch + th) // 2)
        cv2.putText(glyph, self.selected_letter, (tx, ty), self.letter_font, scale,
                    255, self.letter_thickness, cv2.LINE_AA)

        if source_image is not None and source_image.size > 0:
            video = cv2.resize(source_image, (cw, ch), interpolation=cv2.INTER_NEAREST)
        else:
            video = np.zeros((ch, cw, 3), dtype=np.uint8)

        # Original video mapped to panel X/Y and clipped by the letter.
        letter_view = cv2.bitwise_and(video, video, mask=glyph)
        outline = cv2.morphologyEx(glyph, cv2.MORPH_GRADIENT, np.ones((2, 2), np.uint8))
        letter_view = cv2.addWeighted(letter_view, 1.0,
                                      cv2.cvtColor(outline, cv2.COLOR_GRAY2BGR),
                                      0.35, 0)
        panel[top:bottom, left:right] = letter_view

        cv2.drawMarker(panel, (left + cw // 2, top + ch // 2), (110, 110, 110),
                       cv2.MARKER_CROSS, 10, 1)
        return panel

    def compose_display(self, main_frame, source_for_letter):
        panel = self.build_letter_panel(source_for_letter)
        mh, mw = main_frame.shape[:2]
        ph, pw = panel.shape[:2]
        h = max(mh, ph)
        canvas = np.zeros((h, mw + pw, 3), dtype=np.uint8)
        canvas[:mh, :mw] = main_frame
        canvas[:ph, mw:mw + pw] = panel
        cv2.line(canvas, (mw, 0), (mw, h - 1), (90, 90, 90), 2)
        return canvas

    # ==========================================================
    # RUN
    # ==========================================================
    def run(self):

        if not self.cap.isOpened():

            print(
                "Video cannot be opened."
            )

            return

        cv2.namedWindow(
            "Main Video"
        )

        cv2.setMouseCallback(
            "Main Video",
            self.mouse_callback
        )

        print()
        print("======================================")
        print("           VIDEO ANALYZER")
        print("======================================")
        print("Space : Play / Pause")
        print("R     : Select ROI")
        print("D     : Next Frame")
        print("F     : Previous Frame")
        print("1     : Brightness Detection")
        print("2     : Color Detection")
        print("3     : Brightness + Color")
        print("A-Z/a-z: Map video into the typed letter")
        print("+     : Zoom In")
        print("-     : Zoom Out")
        print("M     : Manual Add")
        print("X     : Manual Delete")
        print("P     : Print Coordinates")
        print("Q     : Quit")
        print("======================================")
        print()

        # ======================================================
        # FIRST FRAME
        # ======================================================
        ret, frame = self.cap.read()

        if not ret:

            print(
                "Could not read first frame."
            )

            self.cap.release()
            return

        self.current_frame = frame.copy()
        self.current_frame_idx = 0

        # ======================================================
        # MAIN LOOP
        # ======================================================
        while True:

            # ==================================================
            # READ FRAME
            # ==================================================
            if not self.is_paused:

                ret, frame = self.cap.read()

                if not ret:

                    self.cap.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        0
                    )

                    self.current_frame_idx = 0
                    self.prev_gray = None
                    self.camera_x = 0.0
                    self.camera_y = 0.0
                    self.previous_detections = []

                    continue

                self.current_frame = frame.copy()

                self.current_frame_idx = int(
                    self.cap.get(
                        cv2.CAP_PROP_POS_FRAMES
                    )
                )

            else:

                if self.current_frame is None:
                    continue

                frame = self.current_frame.copy()

            # ==================================================
            # STABILIZATION
            # ==================================================
            stabilized_frame = self.stabilize_frame(
                frame
            )

            display_frame = stabilized_frame.copy()

            # ==================================================
            # ROI
            # ==================================================
            if (
                self.is_roi_selected
                and self.roi is not None
            ):

                x, y, w, h = self.roi

                x1 = max(
                    0,
                    int(x)
                )

                y1 = max(
                    0,
                    int(y)
                )

                x2 = min(
                    stabilized_frame.shape[1],
                    int(x + w)
                )

                y2 = min(
                    stabilized_frame.shape[0],
                    int(y + h)
                )

                if (
                    x2 > x1
                    and y2 > y1
                ):

                    roi = stabilized_frame[
                        y1:y2,
                        x1:x2
                    ].copy()

                    self.current_roi = roi
                    self.current_roi_origin = (
                        x1,
                        y1
                    )

                    # ==========================================
                    # DETECTION
                    # ==========================================
                    auto_objects = self.detect_objects(
                        roi
                    )

                    auto_objects = self.filter_detections(
                        auto_objects
                    )

                    self.detected_objects = []

                    # ==========================================
                    # SAVE DETECTIONS
                    # ==========================================
                    for rx, ry, rw, rh in auto_objects:

                        global_x = int(
                            x1 + rx
                        )

                        global_y = int(
                            y1 + ry
                        )

                        object_data = [
                            global_x,
                            global_y,
                            int(rw),
                            int(rh)
                        ]

                        if self.is_deleted_object(
                            object_data
                        ):
                            continue

                        self.detected_objects.append(
                            object_data
                        )

                    # ==========================================
                    # ROI BLUE BORDER
                    # ==========================================
                    cv2.rectangle(
                        display_frame,
                        (
                            x1,
                            y1
                        ),
                        (
                            x2,
                            y2
                        ),
                        (255, 0, 0),
                        2
                    )

                    # ==========================================
                    # GREEN AUTOMATIC BOXES
                    # ==========================================
                    for gx, gy, gw, gh in (
                        self.detected_objects
                    ):

                        detected_color = self.get_pixel_color(
                            roi,
                            gx - x1,
                            gy - y1,
                            gw,
                            gh
                        )

                        cv2.rectangle(
                            display_frame,
                            (
                                gx,
                                gy
                            ),
                            (
                                gx + gw,
                                gy + gh
                            ),
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            display_frame,
                            detected_color,
                            (gx, max(18, gy - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA
                        )

                    # ==========================================
                    # MANUAL POINTS
                    # ==========================================
                    for px, py in (
                        self.manual_points
                    ):

                        self.draw_manual_point(
                            display_frame,
                            px,
                            py
                        )

                        # No coordinate text on the main video.

                    # ==========================================
                    # MAIN ROI ZOOM
                    # ==========================================
                    if self.main_zoom > 1:

                        zoom_roi = roi.copy()

                        # Draw automatic detections
                        for gx, gy, gw, gh in (
                            self.detected_objects
                        ):

                            local_x = int(
                                gx - x1
                            )

                            local_y = int(
                                gy - y1
                            )

                            cv2.rectangle(
                                zoom_roi,
                                (
                                    local_x,
                                    local_y
                                ),
                                (
                                    local_x + gw,
                                    local_y + gh
                                ),
                                (0, 255, 0),
                                2
                            )

                            detected_color = self.get_pixel_color(
                                roi,
                                local_x,
                                local_y,
                                gw,
                                gh
                            )

                            cv2.putText(
                                zoom_roi,
                                detected_color,
                                (
                                    local_x,
                                    max(
                                        15,
                                        local_y - 5
                                    )
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (0, 255, 0),
                                1,
                                cv2.LINE_AA
                            )

                        # Draw manual points
                        for px, py in (
                            self.manual_points
                        ):

                            self.draw_manual_point(
                                zoom_roi,
                                px,
                                py,
                                x1,
                                y1,
                                1.0
                            )

                        zoomed_roi = cv2.resize(
                            zoom_roi,
                            None,
                            fx=self.main_zoom,
                            fy=self.main_zoom,
                            interpolation=cv2.INTER_NEAREST
                        )

                        zh, zw = zoomed_roi.shape[:2]

                        max_h = display_frame.shape[0]
                        max_w = display_frame.shape[1]

                        final_h = min(
                            zh,
                            max_h
                        )

                        final_w = min(
                            zw,
                            max_w
                        )

                        zoomed_roi = zoomed_roi[
                            :final_h,
                            :final_w
                        ]

                        display_frame[
                            0:final_h,
                            0:final_w
                        ] = zoomed_roi

                        cv2.rectangle(
                            display_frame,
                            (
                                0,
                                0
                            ),
                            (
                                final_w - 1,
                                final_h - 1
                            ),
                            (255, 255, 255),
                            2
                        )

                        cv2.putText(
                            display_frame,
                            f"MAIN ROI ZOOM x{self.main_zoom:.1f}",
                            (
                                10,
                                30
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2
                        )

                    # ==========================================
                    # FROZEN ROI WINDOW
                    # ==========================================
                    if self.frozen_roi is not None:

                        frozen_display = cv2.resize(
                            self.frozen_roi,
                            None,
                            fx=self.frozen_zoom,
                            fy=self.frozen_zoom,
                            interpolation=cv2.INTER_NEAREST
                        )

                        cv2.imshow(
                            "ROI Zoom - FROZEN",
                            frozen_display
                        )

                    # ==========================================
                    # INFO
                    # ==========================================
                    cv2.putText(
                        display_frame,
                        "Mode: "
                        + self.detection_mode,
                        (
                            20,
                            display_frame.shape[0] - 60
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        display_frame,
                        "Detected: "
                        + str(
                            len(
                                self.detected_objects
                            )
                        ),
                        (
                            20,
                            display_frame.shape[0] - 30
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        display_frame,
                        "ROI: "
                        + str(w)
                        + " x "
                        + str(h),
                        (
                            20,
                            90
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    if self.manual_add_mode:

                        cv2.putText(
                            display_frame,
                            "MANUAL ADD: CLICK",
                            (
                                20,
                                120
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

                    elif self.manual_delete_mode:

                        cv2.putText(
                            display_frame,
                            "MANUAL DELETE: CLICK",
                            (
                                20,
                                120
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

            # ==================================================
            # SHOW MAIN VIDEO + LETTER PANEL
            # ==================================================
            if (
                self.is_roi_selected
                and self.current_roi is not None
                and self.current_roi.size > 0
            ):
                letter_source = self.current_roi
            else:
                letter_source = stabilized_frame

            combined_display = self.compose_display(
                display_frame,
                letter_source
            )

            cv2.imshow(
                "Main Video",
                combined_display
            )

            # ==================================================
            # KEYBOARD
            # ==================================================
            key = (
                cv2.waitKey(30)
                & 0xFF
            )

            # ==================================================
            # Q = QUIT
            # ==================================================
            if key == ord("q"):

                print("Exiting...")
                break

            # ==================================================
            # SPACE = PLAY / PAUSE
            # ==================================================
            elif key == ord(" "):

                self.is_paused = (
                    not self.is_paused
                )

                print(
                    "Paused =",
                    self.is_paused
                )

            # ==================================================
            # R = SELECT ROI
            # ==================================================
            elif key == ord("r"):

                self.is_paused = True

                print(
                    "Select ROI with mouse..."
                )

                new_roi = cv2.selectROI(
                    "Main Video",
                    stabilized_frame,
                    False
                )

                if (
                    new_roi[2] > 0
                    and new_roi[3] > 0
                ):

                    self.roi = new_roi
                    self.is_roi_selected = True

                    self.manual_points = []
                    self.deleted_objects = []
                    self.previous_detections = []

                    self.main_zoom = 1.0

                    # Save frozen ROI
                    rx, ry, rw, rh = new_roi

                    rx1 = max(
                        0,
                        int(rx)
                    )

                    ry1 = max(
                        0,
                        int(ry)
                    )

                    rx2 = min(
                        stabilized_frame.shape[1],
                        int(rx + rw)
                    )

                    ry2 = min(
                        stabilized_frame.shape[0],
                        int(ry + rh)
                    )

                    if (
                        rx2 > rx1
                        and ry2 > ry1
                    ):

                        self.frozen_roi = (
                            stabilized_frame[
                                ry1:ry2,
                                rx1:rx2
                            ].copy()
                        )

                    print()
                    print("ROI SELECTED")
                    print("X =", rx)
                    print("Y =", ry)
                    print("Width =", rw)
                    print("Height =", rh)
                    print("Frozen ROI saved.")
                    print()

                self.is_paused = True

            # ==================================================
            # D = NEXT FRAME
            # ==================================================
            elif key == ord("d"):

                self.is_paused = True

                total_frames = int(
                    self.cap.get(
                        cv2.CAP_PROP_FRAME_COUNT
                    )
                )

                self.current_frame_idx = min(
                    self.current_frame_idx + 1,
                    max(0, total_frames - 1)
                )

                self.cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    self.current_frame_idx
                )

                ret, new_frame = self.cap.read()

                if ret:

                    self.current_frame = (
                        new_frame.copy()
                    )

                    self.prev_gray = None

            # ==================================================
            # F = PREVIOUS FRAME
            # ==================================================
            elif key == ord("f"):

                self.is_paused = True

                self.current_frame_idx = max(
                    0,
                    self.current_frame_idx - 1
                )

                self.cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    self.current_frame_idx
                )

                ret, new_frame = self.cap.read()

                if ret:

                    self.current_frame = (
                        new_frame.copy()
                    )

                    self.prev_gray = None

            # ==================================================
            # 1 = BRIGHTNESS
            # ==================================================
            elif key == ord("1"):

                self.detection_mode = "B"
                print("Detection mode = BRIGHTNESS")

            # ==================================================
            # 2 = COLOR
            # ==================================================
            elif key == ord("2"):

                self.detection_mode = "C"
                print("Detection mode = COLOR")

            # ==================================================
            # 3 = BRIGHTNESS + COLOR
            # ==================================================
            elif key == ord("3"):

                self.detection_mode = "A"
                print("Detection mode = BRIGHTNESS + COLOR")

            # ==================================================
            # A-Z / a-z = SELECT LETTER
            # ==================================================
            elif (65 <= key <= 90 or 97 <= key <= 122):

                self.selected_letter = chr(key)
                print("Letter panel =", self.selected_letter)

            # ==================================================
            # + = ZOOM IN
            # ==================================================
            elif key in (
                ord("+"),
                ord("=")
            ):

                if self.is_roi_selected:

                    self.main_zoom += 1

                    if self.main_zoom > 12:
                        self.main_zoom = 12

                    print(
                        "Main ROI Zoom =",
                        self.main_zoom
                    )

            # ==================================================
            # - = ZOOM OUT
            # ==================================================
            elif key in (
                ord("-"),
                ord("_")
            ):

                if self.is_roi_selected:

                    self.main_zoom = max(
                        1,
                        self.main_zoom - 1
                    )

                    print(
                        "Main ROI Zoom =",
                        self.main_zoom
                    )

            # ==================================================
            # M = MANUAL ADD
            # ==================================================
            elif key == ord("m"):

                self.manual_add_mode = (
                    not self.manual_add_mode
                )

                self.manual_delete_mode = False

                print(
                    "Manual Add =",
                    self.manual_add_mode
                )

                if self.manual_add_mode:

                    print(
                        "Click on a missed pixel."
                    )

            # ==================================================
            # X = MANUAL DELETE
            # ==================================================
            elif key == ord("x"):

                self.manual_delete_mode = (
                    not self.manual_delete_mode
                )

                self.manual_add_mode = False

                print(
                    "Manual Delete =",
                    self.manual_delete_mode
                )

                if self.manual_delete_mode:

                    print(
                        "Click on a detected pixel "
                        "or square to delete it."
                    )

            # ==================================================
            # P = PRINT COORDINATES
            # ==================================================
            elif key == ord("p"):

                self.print_coordinates()

        # ==========================================================
        # CLOSE
        # ==========================================================
        self.cap.release()
        cv2.destroyAllWindows()

        print()
        print("Program finished.")


# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":

    analyzer = VideoAnalyzer(
        "night.mp4"
    )

    analyzer.run()
