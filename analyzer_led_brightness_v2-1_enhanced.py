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

        # ==========================================================
        # ROI
        # ==========================================================
        self.roi = None
        self.is_roi_selected = False

        # ==========================================================
        # FROZEN ROI
        # ==========================================================
        self.frozen_roi = None
        self.frozen_roi_origin = None
        self.frozen_detected_objects = []
        self.frozen_zoom = 1.0

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

        # ==========================================================
        # LED / LIGHT-INTENSITY DETECTION
        # ==========================================================
        # B mode detects LEDs from luminance/local brightness only.
        # HSV color is not used, so an LED may change color freely.
        self.led_percentile = 96.0
        self.led_min_gray = 85
        self.led_local_contrast = 6.0
        self.led_peak_radius = 3
        self.led_merge_distance = 3
        self.led_box_size = 8          # Equal green automatic LED square size (px)
        self.green_line_thickness = 1  # Automatic green LED box thickness (px)
        self.red_line_thickness = 2    # Manual red add/delete square thickness (px)
        self.manual_box_size = 10      # Manual red square size (px)
        self.video_zoom = 1.0          # Main video panel zoom for precise M/X editing
        self.led_blur_sigma = 5.0
        self.led_core_percentile = 98.0

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
    # BRIGHTNESS DETECTION
    # ==========================================================
    def detect_bright_pixels(self, image):
        """
        Build a luminance mask for bright LED cores.
        No color/HSV information is required in B mode.
        """
        if image is None or image.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32)

        background = cv2.GaussianBlur(
            gray_f, (0, 0), self.led_blur_sigma
        )
        local_contrast = gray_f - background

        percentile_threshold = float(
            np.percentile(gray_f, self.led_percentile)
        )
        global_threshold = max(
            float(self.led_min_gray),
            min(250.0, percentile_threshold)
        )

        global_mask = gray_f >= global_threshold
        local_mask = (
            (gray_f >= self.led_min_gray) &
            (local_contrast >= self.led_local_contrast)
        )

        mask = np.where(
            global_mask | local_mask, 255, 0
        ).astype(np.uint8)

        # Minimal cleanup; heavy morphology would merge nearby LEDs.
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)
        )
        return mask

    def _detect_led_centers(self, image):
        """High-recall LED center detection using multi-scale local contrast."""
        if image is None or image.size == 0:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32)
        bg_small = cv2.GaussianBlur(gray_f, (0, 0), self.led_small_blur_sigma)
        bg_large = cv2.GaussianBlur(gray_f, (0, 0), self.led_blur_sigma)
        contrast_small = gray_f - bg_small
        contrast_large = gray_f - bg_large
        score = (0.70 * np.maximum(contrast_small, 0.0) +
                 0.30 * np.maximum(contrast_large, 0.0))

        percentile = float(np.percentile(gray_f, self.led_percentile))
        intensity_threshold = max(float(self.led_min_gray), min(245.0, percentile))
        radius = max(1, int(self.led_peak_radius))
        local_max = cv2.dilate(gray, np.ones((2*radius+1, 2*radius+1), np.uint8))
        is_peak = gray_f >= (local_max.astype(np.float32) - 1.0)
        score_percentile = float(np.percentile(score, 90.0))
        score_threshold = max(float(self.led_min_score), min(35.0, score_percentile*0.55))

        candidate = is_peak & (
            (score >= score_threshold) |
            ((gray_f >= intensity_threshold) &
             (contrast_large >= self.led_local_contrast*0.65)) |
            ((gray_f >= float(self.led_min_gray)+self.led_bright_bonus) &
             (contrast_small >= self.led_local_contrast*0.45))
        )

        # Direct peak extraction prevents connected-components from merging
        # adjacent LEDs that have equal/near-equal brightness.
        ys, xs = np.where(candidate)
        candidates = []
        for iy, ix in zip(ys.tolist(), xs.tolist()):
            brightness = float(gray_f[iy, ix])
            local_score = float(score[iy, ix])
            local_c = float(contrast_large[iy, ix])
            final_score = (local_score +
                           0.08*max(0.0, brightness-self.led_min_gray) +
                           0.35*max(0.0, local_c))
            if (final_score >= self.led_min_score or
                (brightness >= intensity_threshold and
                 local_c >= self.led_local_contrast*0.55)):
                candidates.append((final_score, brightness, local_score, int(ix), int(iy)))

        candidates.sort(key=lambda item:(item[0], item[1], item[2]), reverse=True)
        centers=[]
        merge_d=max(2,int(self.led_merge_distance))
        merge_d2=merge_d*merge_d
        for final_score, brightness, local_score, cx, cy in candidates:
            if any((cx-ox)**2+(cy-oy)**2 <= merge_d2 for ox,oy in centers):
                continue
            centers.append((cx,cy))
        return centers

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

        # ======================================================
        # LED INTENSITY MODE
        # ======================================================
        # B is now a true LED detector: it finds local luminance peaks
        # and returns equal-sized squares centered on those peaks.
        if self.detection_mode == "B":
            centers = self._detect_led_centers(image)

            size = max(2, int(self.led_box_size))
            half = size // 2
            ih, iw = image.shape[:2]

            objects = []
            for cx, cy in centers:
                x = int(cx - half)
                y = int(cy - half)

                x = max(0, min(x, max(0, iw - size)))
                y = max(0, min(y, max(0, ih - size)))

                objects.append([x, y, size, size])

            return objects

        # ======================================================
        # LEGACY COLOR / COMBINED MODES (preserved)
        # ======================================================
        bright_mask = self.detect_bright_pixels(image)
        color_mask = self.detect_color(image)

        if self.detection_mode == "C":
            mask = color_mask
        elif self.detection_mode == "A":
            mask = cv2.bitwise_or(bright_mask, color_mask)
        else:
            mask = bright_mask

        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, self.morph_kernel
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, self.morph_kernel
        )

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
        )

        objects = []
        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])

            if area < self.min_area or area > self.max_area:
                continue

            objects.append([x, y, w, h])

        return objects

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
        # Manual editing is performed only from the MAIN dashboard panel.
        main_x = getattr(self, "dashboard_main_x", 0)
        main_y = getattr(self, "dashboard_main_y", 0)
        main_w = getattr(self, "dashboard_main_w", 1)
        main_h = getattr(self, "dashboard_main_h", 1)

        if not (main_x <= x < main_x + main_w and main_y <= y < main_y + main_h):
            return -1, -1

        frame = self.current_frame
        if frame is None:
            return -1, -1

        fh, fw = frame.shape[:2]
        scale = min(main_w / fw, main_h / fh)
        shown_w = max(1, int(fw * scale))
        shown_h = max(1, int(fh * scale))
        ox = main_x + (main_w - shown_w) // 2
        oy = main_y + (main_h - shown_h) // 2

        if not (ox <= x < ox + shown_w and oy <= y < oy + shown_h):
            return -1, -1

        vx = (x - ox) / scale
        vy = (y - oy) / scale

        # The displayed main video is center-cropped after video zoom.
        if self.video_zoom > 1.0:
            fh, fw = frame.shape[:2]
            crop_x = (fw * self.video_zoom - fw) / 2.0
            crop_y = (fh * self.video_zoom - fh) / 2.0
            vx = (vx + crop_x) / self.video_zoom
            vy = (vy + crop_y) / self.video_zoom

        return int(vx), int(vy)

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

        if x < 0 or y < 0:
            return

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

                # Keep the frozen Zoom panel synchronized with manual deletion.
                kept_frozen = []
                for frozen_obj in self.frozen_detected_objects:
                    fx, fy, fw, fh = frozen_obj
                    fcx = fx + fw / 2
                    fcy = fy + fh / 2
                    if ((fcx - x) ** 2 + (fcy - y) ** 2) > 900:
                        kept_frozen.append(frozen_obj)
                self.frozen_detected_objects = kept_frozen

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
            int((self.led_box_size / 2) * scale)
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

        if self.roi is not None:
            rx,ry,rw,rh=map(int,self.roi)
            print()
            print("ROI BOUNDARY PIXELS (GLOBAL VIDEO COORDINATES):")
            print("TOP-LEFT     -> X =",rx,"Y =",ry)
            print("TOP-RIGHT    -> X =",rx+rw-1,"Y =",ry)
            print("BOTTOM-LEFT  -> X =",rx,"Y =",ry+rh-1)
            print("BOTTOM-RIGHT -> X =",rx+rw-1,"Y =",ry+rh-1)
            print("ROI SIZE     -> W =",rw,"H =",rh)

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
                1
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
    # RUN
    # ==========================================================
    def _fit_image(self, image, width, height, interpolation=cv2.INTER_AREA):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        if image is None or image.size == 0:
            return canvas
        ih, iw = image.shape[:2]
        scale = min(width / iw, height / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=interpolation)
        ox, oy = (width - nw) // 2, (height - nh) // 2
        canvas[oy:oy+nh, ox:ox+nw] = resized
        return canvas

    def _zoom_main_video(self, frame):
        if frame is None or frame.size == 0 or self.video_zoom <= 1.0:
            return frame.copy() if frame is not None else frame

        h, w = frame.shape[:2]
        scale = float(self.video_zoom)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        enlarged = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_NEAREST)

        # Keep the center fixed while zooming, so M/X clicks target the
        # same video coordinates with much greater precision.
        sx = max(0, (nw - w) // 2)
        sy = max(0, (nh - h) // 2)
        return enlarged[sy:sy+h, sx:sx+w].copy()

    def _draw_main_boxes(self, panel):
        if self.current_frame is None:
            return
        ph, pw = panel.shape[:2]
        fh, fw = self.current_frame.shape[:2]
        scale = min(pw / fw, ph / fh)
        nw, nh = max(1, int(fw*scale)), max(1, int(fh*scale))
        ox, oy = (pw-nw)//2, (ph-nh)//2

        for gx, gy, gw, gh in self.detected_objects:
            x1 = ox + int(gx*scale)
            y1 = oy + int(gy*scale)
            x2 = ox + int((gx+gw)*scale)
            y2 = oy + int((gy+gh)*scale)
            x2 = max(x2, x1+6)
            y2 = max(y2, y1+6)
            cv2.rectangle(panel, (x1,y1), (x2,y2), (0,255,0), 1)

        for px, py in self.manual_points:
            cx, cy = ox + int(px*scale), oy + int(py*scale)
            s = max(5, int(6*max(scale,1)))
            cv2.rectangle(panel, (cx-s,cy-s), (cx+s,cy+s), (0,0,255), 2)

    def _make_map_panel(self, width, height):
        """Clean map: only green automatic and red manual squares."""
        panel=np.zeros((height,width,3),dtype=np.uint8)
        if self.roi is None:
            return panel
        rx,ry,rw,rh=map(int,self.roi)
        margin=35
        scale=min((width-2*margin)/max(1,rw),(height-2*margin)/max(1,rh))
        aw,ah=max(1,int(round(rw*scale))),max(1,int(round(rh*scale)))
        ox,oy=(width-aw)//2,(height-ah)//2
        for gx,gy,gw,gh in self.detected_objects:
            lx,ly=gx-rx,gy-ry
            cx=ox+int(round((lx+gw/2.0)*scale)); cy=oy+int(round((ly+gh/2.0)*scale))
            bw,bh=max(4,int(round(gw*scale))),max(4,int(round(gh*scale)))
            cv2.rectangle(panel,(cx-bw//2,cy-bh//2),(cx+bw//2,cy+bh//2),(0,255,0),self.green_line_thickness)
        for px,py in self.manual_points:
            cx=ox+int(round((px-rx)*scale)); cy=oy+int(round((py-ry)*scale))
            s=max(4,int(round(self.manual_box_size*scale/2.0)))
            cv2.rectangle(panel,(cx-s,cy-s),(cx+s,cy+s),(0,0,255),self.red_line_thickness)
        return panel

    def _make_coordinates_panel(self,width,height):
        """Bottom-left: exact ROI corner pixels plus all visible LED X/Y coordinates."""
        panel=np.zeros((height,width,3),dtype=np.uint8)
        cv2.putText(panel,"COORDINATES",(18,28),cv2.FONT_HERSHEY_SIMPLEX,.70,(255,255,255),2)
        if self.roi is None:
            cv2.putText(panel,"Press R and select the LED area",(18,62),cv2.FONT_HERSHEY_SIMPLEX,.55,(190,190,190),1,cv2.LINE_AA)
            return panel
        rx,ry,rw,rh=map(int,self.roi); xl=rx+rw-1; yt=ry+rh-1
        cv2.putText(panel,f"ROI: W={rw}  H={rh}",(18,57),cv2.FONT_HERSHEY_SIMPLEX,.52,(210,210,210),1,cv2.LINE_AA)
        corners=[("TOP-LEFT",rx,ry),("TOP-RIGHT",xl,ry),("BOTTOM-LEFT",rx,yt),("BOTTOM-RIGHT",xl,yt)]
        y=88
        for name,x,yy in corners:
            cv2.putText(panel,f"{name}: X={x}  Y={yy}",(18,y),cv2.FONT_HERSHEY_SIMPLEX,.50,(255,255,255),1,cv2.LINE_AA); y+=24
        y+=4
        cv2.line(panel,(18,y-10),(width-18,y-10),(70,70,70),1)
        cv2.putText(panel,f"AUTO: {len(self.detected_objects)}   MANUAL: {len(self.manual_points)}",(18,y+12),cv2.FONT_HERSHEY_SIMPLEX,.48,(210,210,210),1,cv2.LINE_AA)
        y+=38
        rows=[(f"A{i:03d}: X={int(gx)}  Y={int(gy)}",(0,255,0)) for i,(gx,gy,gw,gh) in enumerate(self.detected_objects,1)]
        rows += [(f"M{i:03d}: X={int(px)}  Y={int(py)}",(0,0,255)) for i,(px,py) in enumerate(self.manual_points,1)]
        col_w=max(300,width//2); max_rows=max(1,(height-y-10)//20); max_visible=max_rows*2; hidden=max(0,len(rows)-max_visible)
        for idx,(line,col) in enumerate(rows[:max_visible]):
            col_i=idx//max_rows; row=idx%max_rows
            cv2.putText(panel,line,(18+col_i*col_w,y+row*20),cv2.FONT_HERSHEY_SIMPLEX,.44,col,1,cv2.LINE_AA)
        if hidden:
            cv2.putText(panel,f"... {hidden} more; press P for all",(18,height-8),cv2.FONT_HERSHEY_SIMPLEX,.44,(170,170,170),1,cv2.LINE_AA)
        return panel

    def _compose_dashboard(self, display_frame):
        W,H,G=800,450,8
        dashboard=np.zeros((H*2+G,W*2+G,3),dtype=np.uint8)

        zoomed_display = self._zoom_main_video(display_frame)
        main=self._fit_image(zoomed_display,W,H,cv2.INTER_NEAREST)
        self.dashboard_main_x,self.dashboard_main_y=0,0
        self.dashboard_main_w,self.dashboard_main_h=W,H
        self._draw_main_boxes(main)
        cv2.rectangle(main,(0,0),(W-1,H-1),(255,255,255),2)
        cv2.rectangle(main,(0,0),(W,42),(0,0,0),-1)
        cv2.putText(main,"MAIN VIDEO",(15,29),cv2.FONT_HERSHEY_SIMPLEX,.72,(255,255,255),2)
        cv2.putText(main,f"Mode:{self.detection_mode}  Detected:{len(self.detected_objects)}  VideoZoom:x{self.video_zoom:.1f}  ROI-Zoom:x{self.main_zoom:.1f}",
                    (190,29),cv2.FONT_HERSHEY_SIMPLEX,.48,(210,210,210),1)
        if self.manual_add_mode:
            cv2.putText(main,"ADD - click",(650,29),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
        elif self.manual_delete_mode:
            cv2.putText(main,"DELETE - click",(620,29),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)

        # ----------------------------------------------------------
        # ZOOM PANEL
        # IMPORTANT: do not resize the zoomed image back down with
        # _fit_image(), otherwise + / - appears to do nothing.
        # At x1 the ROI fills the panel; higher values magnify it and
        # the center is shown/cropped so individual pixels remain clear.
        # ----------------------------------------------------------
        # ----------------------------------------------------------
        # ZOOM PANEL
        # The Zoom panel is intentionally FROZEN. It uses the ROI
        # snapshot captured when R/ROI selection was completed, so
        # the image does not move while the video plays. + / - only
        # changes magnification, making it easy to inspect exactly
        # which pixels were detected.
        # ----------------------------------------------------------
        if self.frozen_roi is not None and self.frozen_roi.size > 0:
            zsrc = self.frozen_roi.copy()
            rx, ry = self.frozen_roi_origin

            # Draw the detections belonging to the frozen frame.
            for gx, gy, gw, gh in self.frozen_detected_objects:
                lx, ly = int(gx - rx), int(gy - ry)
                if lx + gw < 0 or ly + gh < 0 or lx >= zsrc.shape[1] or ly >= zsrc.shape[0]:
                    continue
                cv2.rectangle(
                    zsrc,
                    (lx, ly),
                    (lx + max(int(gw), 4), ly + max(int(gh), 4)),
                    (0, 255, 0),
                    1
                )

            # Manual points are current/global coordinates, so they are
            # also drawn on the frozen image for easy inspection.
            for px, py in self.manual_points:
                self.draw_manual_point(zsrc, px, py, rx, ry, 1.0)

            zh, zw = zsrc.shape[:2]

            # x1 fits the frozen ROI into the panel. Higher values
            # genuinely magnify it and crop around the center.
            base_scale = min(W / max(1, zw), H / max(1, zh))
            scale = max(0.05, base_scale * self.main_zoom)

            nw = max(1, int(round(zw * scale)))
            nh = max(1, int(round(zh * scale)))

            enlarged = cv2.resize(
                zsrc,
                (nw, nh),
                interpolation=cv2.INTER_NEAREST
            )

            zoom = np.zeros((H, W, 3), dtype=np.uint8)

            if nw <= W:
                dx = (W - nw) // 2
                sx1, sx2 = 0, nw
                dx1, dx2 = dx, dx + nw
            else:
                sx1 = (nw - W) // 2
                sx2 = sx1 + W
                dx1, dx2 = 0, W

            if nh <= H:
                dy = (H - nh) // 2
                sy1, sy2 = 0, nh
                dy1, dy2 = dy, dy + nh
            else:
                sy1 = (nh - H) // 2
                sy2 = sy1 + H
                dy1, dy2 = 0, H

            zoom[dy1:dy2, dx1:dx2] = enlarged[sy1:sy2, sx1:sx2]

        else:
            zoom = np.zeros((H, W, 3), dtype=np.uint8)
            cv2.putText(
                zoom,
                "Select ROI first",
                (25, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                .75,
                (200, 200, 200),
                2
            )

        cv2.rectangle(zoom, (0, 0), (W-1, H-1), (255, 255, 255), 2)
        cv2.putText(
            zoom,
            f"ZOOM PANEL - FROZEN x{self.main_zoom:.1f}   (+ / -)",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            .65,
            (255, 255, 255),
            2
        )

        # Bottom-left = coordinates; bottom-right = clean LED map.
        cp=self._make_coordinates_panel(W,H)
        mp=self._make_map_panel(W,H)
        cv2.rectangle(cp,(0,0),(W-1,H-1),(255,255,255),2)
        cv2.rectangle(mp,(0,0),(W-1,H-1),(255,255,255),2)

        dashboard[:H,:W]=main
        dashboard[:H,W+G:W*2+G]=zoom
        dashboard[H+G:H*2+G,:W]=cp
        dashboard[H+G:H*2+G,W+G:W*2+G]=mp
        return dashboard

    def run(self):

        if not self.cap.isOpened():
            print("Video cannot be opened.")
            return

        window_name="Video Analyzer - 4 Panel"
        cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name,1600,908)
        cv2.setMouseCallback(window_name,self.mouse_callback)

        print("======================================")
        print("        VIDEO ANALYZER - 4 PANELS")
        print("======================================")
        print("Space Play/Pause | R ROI | D Next | F Previous")
        print("B LED Brightness/Intensity | C Color | A Brightness+Color")
        print("+/- ROI Zoom | [/] Main Video Zoom | M Manual Add | X Manual Delete")
        print("P Print Coordinates | Q Quit")
        print("Manual M/X clicks are done on MAIN panel.")
        print("LED detector: luminance/local peaks | Fixed square:", self.led_box_size)
        print("======================================")

        ret,frame=self.cap.read()
        if not ret:
            print("Could not read first frame.")
            self.cap.release()
            return
        self.current_frame=frame.copy()
        self.current_frame_idx=0

        while True:
            if not self.is_paused:
                ret,frame=self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                    self.current_frame_idx=0
                    self.prev_gray=None
                    self.camera_x=self.camera_y=0.0
                    self.previous_detections=[]
                    continue
                self.current_frame=frame.copy()
                self.current_frame_idx=int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            elif self.current_frame is None:
                continue
            else:
                frame=self.current_frame.copy()

            stabilized_frame=self.stabilize_frame(frame)
            display_frame=stabilized_frame.copy()

            if self.is_roi_selected and self.roi is not None:
                x,y,w,h=self.roi
                x1,y1=max(0,int(x)),max(0,int(y))
                x2=min(stabilized_frame.shape[1],int(x+w))
                y2=min(stabilized_frame.shape[0],int(y+h))

                if x2>x1 and y2>y1:
                    roi=stabilized_frame[y1:y2,x1:x2].copy()
                    self.current_roi=roi
                    self.current_roi_origin=(x1,y1)

                    auto=self.filter_detections(self.detect_objects(roi))
                    self.detected_objects=[]
                    for rx,ry,rw,rh in auto:
                        obj=[int(x1+rx),int(y1+ry),int(rw),int(rh)]
                        if not self.is_deleted_object(obj):
                            self.detected_objects.append(obj)

                    cv2.rectangle(display_frame,(x1,y1),(x2,y2),(255,0,0),2)
                    for gx,gy,gw,gh in self.detected_objects:
                        cv2.rectangle(display_frame,(gx,gy),(gx+max(gw,4),gy+max(gh,4)),(0,255,0),1)
                    for px,py in self.manual_points:
                        self.draw_manual_point(display_frame,px,py)
            else:
                self.current_roi=None
                self.current_roi_origin=None
                self.detected_objects=[]
                self.frozen_roi=None
                self.frozen_roi_origin=None
                self.frozen_detected_objects=[]

            dashboard=self._compose_dashboard(display_frame)
            cv2.imshow(window_name,dashboard)
            key=cv2.waitKey(30)&0xFF

            if key==ord("q"):
                break

            elif key==ord(" "):
                self.is_paused=not self.is_paused

            elif key==ord("r"):
                self.is_paused=True
                select_name="Select ROI"
                cv2.namedWindow(select_name,cv2.WINDOW_NORMAL)
                cv2.resizeWindow(select_name,1000,650)
                new_roi=cv2.selectROI(select_name,stabilized_frame,False)
                cv2.destroyWindow(select_name)
                if new_roi[2]>0 and new_roi[3]>0:
                    self.roi=new_roi
                    self.is_roi_selected=True
                    self.manual_points=[]
                    self.deleted_objects=[]
                    self.previous_detections=[]
                    self.main_zoom=1.0
                    self.video_zoom=1.0
                    self.frozen_roi=None
                    self.frozen_roi_origin=None
                    self.frozen_detected_objects=[]
                    rx,ry,rw,rh=new_roi
                    rx1,ry1=max(0,int(rx)),max(0,int(ry))
                    rx2=min(stabilized_frame.shape[1],int(rx+rw))
                    ry2=min(stabilized_frame.shape[0],int(ry+rh))
                    if rx2>rx1 and ry2>ry1:
                        # Capture a permanent snapshot for the Zoom panel.
                        # It will not change when the video continues playing.
                        self.frozen_roi=stabilized_frame[ry1:ry2,rx1:rx2].copy()
                        self.frozen_roi_origin=(rx1,ry1)

                        frozen_auto=self.detect_objects(self.frozen_roi)
                        self.frozen_detected_objects=[]
                        for frx,fry,frw,frh in frozen_auto:
                            fobj=[int(rx1+frx),int(ry1+fry),int(frw),int(frh)]
                            if not self.is_deleted_object(fobj):
                                self.frozen_detected_objects.append(fobj)

            elif key==ord("d"):
                self.is_paused=True
                total=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.current_frame_idx=min(self.current_frame_idx+1,max(0,total-1))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.current_frame_idx)
                ret,nf=self.cap.read()
                if ret:
                    self.current_frame=nf.copy()
                    self.prev_gray=None

            elif key==ord("f"):
                self.is_paused=True
                self.current_frame_idx=max(0,self.current_frame_idx-1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.current_frame_idx)
                ret,nf=self.cap.read()
                if ret:
                    self.current_frame=nf.copy()
                    self.prev_gray=None

            elif key==ord("b"):
                self.detection_mode="B"
            elif key==ord("c"):
                self.detection_mode="C"
            elif key==ord("a"):
                self.detection_mode="A"

            elif key in (ord("+"),ord("=")):
                if self.is_roi_selected:
                    self.main_zoom=min(12,self.main_zoom+1)

            elif key in (ord("-"),ord("_")):
                if self.is_roi_selected:
                    self.main_zoom=max(1,self.main_zoom-1)

            elif key in (ord("]"), ord("}")):
                self.video_zoom = min(8.0, self.video_zoom + 0.5)

            elif key in (ord("["), ord("{")):
                self.video_zoom = max(1.0, self.video_zoom - 0.5)

            elif key==ord("m"):
                self.manual_add_mode=not self.manual_add_mode
                self.manual_delete_mode=False
                print("Manual Add =",self.manual_add_mode)

            elif key==ord("x"):
                self.manual_delete_mode=not self.manual_delete_mode
                self.manual_add_mode=False
                print("Manual Delete =",self.manual_delete_mode)

            elif key==ord("p"):
                self.print_coordinates()

        self.cap.release()
        cv2.destroyAllWindows()
        print("Program finished.")


# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":

    analyzer = VideoAnalyzer(
        "night.mp4"
    )

    analyzer.run()
