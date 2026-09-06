import cv2
import numpy as np

# --- تنظیمات دوربین و متغیرها ---
cap = cv2.VideoCapture(0)
roi_rect = None 
start_point = None
is_selecting = False
is_locked = False 

# --- تنظیمات رنگ (دقیقاً مطابق تنظیمات شما) ---
lower_blue = np.array([100, 120, 70]) 
upper_blue = np.array([130, 255, 255])
kernel = np.ones((5, 5), np.uint8)

# --- متغیرهای لرزه‌گیری (Smoothing) ---
# alpha: هرچه کمتر باشد، حرکت نرم‌تر اما با تأخیر بیشتر (اگر 1 باشد لرزش داریم)
alpha = 0.3 

def mouse_callback(event, x, y, flags, param):
    global start_point, is_selecting, roi_rect, is_locked
    if event == cv2.EVENT_LBUTTONDOWN:
        is_selecting = True
        start_point = (x, y)
        is_locked = False 
    elif event == cv2.EVENT_LBUTTONUP:
        is_selecting = False
        x1, y1 = start_point
        x2, y2 = x, y
        # تعیین کادر اولیه بر اساس انتخاب موس
        roi_rect = (min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2))
        is_locked = True 

cv2.namedWindow('Smart Selection')
cv2.setMouseCallback('Smart Selection', mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret: 
        break
        
    display_frame = frame.copy()
    h_img, w_img = frame.shape[:2]

    # --- ۱. پردازش رنگ و ایجاد ماسک (دقیقاً مطابق استانداردهای شما) ---
    blurred = cv2.medianBlur(frame, 5)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # عملیات مورفولوژیک برای حذف نویز و پر کردن حفره‌ها
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # --- ۲. مدیریت کادر متحرک و لرزه‌گیری ---
    if roi_rect:
        rx, ry, rw, rh = roi_rect
        
        if is_locked:
            # تعریف محدوده جستجو (کمی بزرگتر از کادر فعلی)
            margin = 40
            x_s = max(0, rx - margin)
            y_s = max(0, ry - margin)
            x_e = min(w_img, rx + rw + margin)
            y_e = min(h_img, ry + rh + margin)

            # استخراج ماسک در ناحیه کادر
            roi_mask = mask[y_s:y_e, x_s:x_e]
            contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # پیدا کردن بزرگترین کانتور آبی در محدوده کادر
                largest_cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_cnt) > 300: # حداقل اندازه برای تشخیص
                    bx, by, bw, bh = cv2.boundingRect(largest_cnt)
                    
                    # مختصات هدف (جایی که جسم الان هست)
                    target_rx = x_s + bx
                    target_ry = y_s + by
                    
                    # --- اعمال لرزه‌گیری (Low-pass Filter) ---
                    # به جای پریدن مستقیم، مختصات جدید را با وزن کم (alpha) اضافه می‌کنیم
                    new_rx = int(rx + alpha * (target_rx - rx))
                    new_ry = int(ry + alpha * (target_ry - ry))
                    
                    # جلوگیری از خروج کادر از مرز تصویر
                    new_rx = max(0, min(new_rx, w_img - rw))
                    new_ry = max(0, min(new_ry, h_img - rh))
                    
                    roi_rect = (new_rx, new_ry, rw, rh)
                    # به‌روزرسانی متغیرهای محلی برای رسم در فریم فعلی
                    rx, ry, rw, rh = roi_rect

        # --- ۳. رسم خروجی‌ها ---
        # رسم کادر زرد (انتخاب کاربر)
        cv2.rectangle(display_frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
        
        # نمایش مختصات لحظه‌ای (به‌روز شده و لرزش‌گیری شده)
        coord_text = f"X:{rx} Y:{ry} W:{rw} H:{rh}"
        cv2.putText(display_frame, coord_text, (rx, ry - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # نمایش پنجره‌ها
    cv2.imshow('Blue Mask Map', mask)
    cv2.imshow('Smart Selection', display_frame)

    # کنترل‌های کیبورد
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'): # Reset
        roi_rect = None
        is_locked = False
    elif key == ord('q'): # Quit
        break

cap.release()
cv2.destroyAllWindows()