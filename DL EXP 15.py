import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image (paste your image path here)
img = cv2.imread(r"C:\Users\K Pranitha Reddy\Downloads\EXP 15.jpg")

# Check if image loaded
if img is None:
    print("Error: Image not found. Check the path.")
else:
    
    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape image to pixel values
    pixel_values = rgb_img.reshape((-1,3))
    pixel_values = np.float32(pixel_values)

    # Define criteria for K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100,0.2)

    # Number of clusters
    k = 3

    # Apply K-means
    _, labels, centers = cv2.kmeans(pixel_values,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)

    segmented_data = centers[labels.flatten()]

    segmented_image = segmented_data.reshape(rgb_img.shape)

    # Display results
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(rgb_img)
    plt.title("Original Cat Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(segmented_image)
    plt.title("Segmented Image")
    plt.axis("off")

    plt.show()
