
import sys
from PIL import Image
import matplotlib.pyplot as plt

def main():
    # בדיקה שהועבר שם קובץ
    if len(sys.argv) != 2:
        print("Usage: python ex01_01.py image.jpg")
        return

    image_path = sys.argv[1]

    # קריאת התמונה
    img = Image.open(image_path)

    # פירוק לערוצי צבע
    r, g, b = img.split()

    # הצגת הערוצים
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(r, cmap='Reds')
    plt.title("Red Channel")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(g, cmap='Greens')
    plt.title("Green Channel")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(b, cmap='Blues')
    plt.title("Blue Channel")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()