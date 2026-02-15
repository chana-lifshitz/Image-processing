import sys
import cv2
import numpy as np
import colorsys

# קלט משורת הפקודה
R = int(sys.argv[1])
G = int(sys.argv[2])
B = int(sys.argv[3])

# ---------- א. מימוש ישיר של נוסחאות ----------

# נרמול
r, g, b = R/255.0, G/255.0, B/255.0

# HSV (נוסחאות סטנדרטיות)
hsv_manual = colorsys.rgb_to_hsv(r, g, b)  # H,S,V בטווח [0,1]

# HSL
hsl_manual = colorsys.rgb_to_hls(r, g, b)  # H,L,S בטווח [0,1]

# YCrCb (BT.601)
Y  = 0.299*R + 0.587*G + 0.114*B
Cr = 128 + 0.5*R - 0.418688*G - 0.081312*B
Cb = 128 - 0.168736*R - 0.331264*G + 0.5*B
ycrcb_manual = (Y, Cr, Cb)

# ---------- ב. שימוש ב־cv2.cvtColor ----------

img = np.uint8([[[B, G, R]]])  # OpenCV עובד בפורמט BGR

hsv_cv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[0][0]
hsl_cv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[0][0]
ycrcb_cv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[0][0]

# ---------- הדפסה ----------

print("HSV manual:", hsv_manual)
print("HSV cv2   :", hsv_cv)

print("HSL manual:", hsl_manual)
print("HSL cv2   :", hsl_cv)

print("YCrCb manual:", ycrcb_manual)
print("YCrCb cv2   :", ycrcb_cv)


