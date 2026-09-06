import cv2
import numpy as np

# =============================
# تنظیمات
# =============================

VIDEO_PATH = "rang.mp4"

# اندازه Map
MAP_WIDTH = 800
MAP_HEIGHT = 800

# اندازه هر پیکسل/مربع در Map
PIXEL_SIZE = 8


# =============================
# باز کردن ویدیو
# =============================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ ویدیو باز نشد!")
    exit()

print("✅ ویدیو باز شد!")


# =============================
# حلقه اصلی
# =============================

while True:

    ret, frame = cap.read()

    if not ret:
        print("پایان ویدیو")
        break

    height, width = frame.shape[:2]


    # =============================
    # تشخیص رنگ آبی
    # =============================

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 80, 50])
    upper_blue = np.array([135, 255, 255])

    blue_mask = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )


    # =============================
    # حذف نویز
    # =============================

    kernel = np.ones((5, 5), np.uint8)

    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # =============================
    # ساخت Map پیکسلی
    # =============================

    MAP_COLS = MAP_WIDTH // PIXEL_SIZE
    MAP_ROWS = MAP_HEIGHT // PIXEL_SIZE

    small_mask = cv2.resize(
        blue_mask,
        (MAP_COLS, MAP_ROWS),
        interpolation=cv2.INTER_AREA
    )


    # Map سیاه

    pixel_map_small = np.zeros(
        (MAP_ROWS, MAP_COLS, 3),
        dtype=np.uint8
    )


    # قسمت‌های آبی

    pixel_map_small[
        small_mask > 40
    ] = (255, 0, 0)


    # =============================
    # بزرگ کردن پیکسل‌ها
    # =============================

    pixel_map = cv2.resize(
        pixel_map_small,
        (MAP_WIDTH, MAP_HEIGHT),
        interpolation=cv2.INTER_NEAREST
    )


    # =============================
    # شبکه Map
    # =============================

    for x in range(
        0,
        MAP_WIDTH,
        PIXEL_SIZE
    ):

        cv2.line(
            pixel_map,
            (x, 0),
            (x, MAP_HEIGHT),
            (35, 35, 35),
            1
        )


    for y in range(
        0,
        MAP_HEIGHT,
        PIXEL_SIZE
    ):

        cv2.line(
            pixel_map,
            (0, y),
            (MAP_WIDTH, y),
            (35, 35, 35),
            1
        )


    # =============================
    # عنوان Map
    # =============================

    cv2.putText(
        pixel_map,
        "BLUE PIXEL MAP",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # =============================
    # بزرگ کردن ویدیو و Map برای نمایش
    # =============================

    DISPLAY_HEIGHT = 500

    video_ratio = width / height

    video_display_width = int(
        DISPLAY_HEIGHT * video_ratio
    )

    frame_display = cv2.resize(
        frame,
        (
            video_display_width,
            DISPLAY_HEIGHT
        )
    )


    map_ratio = MAP_WIDTH / MAP_HEIGHT

    map_display_width = int(
        DISPLAY_HEIGHT * map_ratio
    )

    map_display = cv2.resize(
        pixel_map,
        (
            map_display_width,
            DISPLAY_HEIGHT
        ),
        interpolation=cv2.INTER_NEAREST
    )


    # =============================
    # کنار هم قرار دادن
    # =============================

    combined = np.hstack(
        (
            frame_display,
            map_display
        )
    )


    # =============================
    # نمایش
    # =============================

    cv2.imshow(
        "VIDEO + BLUE PIXEL MAP",
        combined
    )


    # =============================
    # خروج با Q
    # =============================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =============================
# پایان
# =============================

cap.release()

cv2.destroyAllWindows()