import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_edge_detection(image_path, output_path, low_threshold, high_threshold):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Failed to read the image from {image_path}")
        return

    # Apply Canny edge detector
    edges = cv2.Canny(image, low_threshold, high_threshold)

    # Save the Canny edge-detected image
    cv2.imwrite(output_path, edges)
    print(f"Canny edge-detected image saved as {output_path}")

    # Plot the original and Canny edge-detected images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Detected Image')
    plt.axis('off')

    plt.show()

    # Display the images
    cv2.imshow('Original Image', image)
    cv2.imshow('Canny Edge Detected Image', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
input_image_path = 'apple.jpeg'
output_image_path = 'canny_edge_detected_image.jpg'
low_threshold = 50  # Lower bound for the hysteresis thresholding
high_threshold = 150  # Upper bound for the hysteresis thresholding

canny_edge_detection(input_image_path, output_image_path, low_threshold, high_threshold)
