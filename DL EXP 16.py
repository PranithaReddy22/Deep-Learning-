import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image (paste your image path here)
img = cv2.imread(r"C:\Users\K Pranitha Reddy\Downloads\images.jpg")

# Check if image loaded
if img is None:
    print("Error: Image not found. Check the path.")
else:
    
    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu Threshold
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Dilation
    dilation = cv2.dilate(closing,kernel,iterations=2)

    # Display results
    plt.figure(figsize=(12,8))

    plt.subplot(231)
    plt.imshow(rgb_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(232)
    plt.imshow(gray,cmap='gray')
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.subplot(233)
    plt.imshow(thresh,cmap='gray')
    plt.title("Otsu Threshold")
    plt.axis("off")

    plt.subplot(234)
    plt.imshow(closing,cmap='gray')
    plt.title("Morphological Closing")
    plt.axis("off")

    plt.subplot(235)
    plt.imshow(dilation,cmap='gray')
    plt.title("Dilation")
    plt.axis("off")

    plt.show()
