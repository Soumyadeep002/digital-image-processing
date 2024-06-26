import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image_path, output_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Failed to read the image from {image_path}")
        return

    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(image)

    # Save the equalized image
    cv2.imwrite(output_path, equalized_image)
    print(f"Equalized image saved as {output_path}")

    # Plot the original and equalized histograms
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(image.flatten(), 256, [0, 256], color='r')
    plt.title('Histogram of Original Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(equalized_image.flatten(), 256, [0, 256], color='r')
    plt.title('Histogram of Equalized Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.show()

    # Display the original and equalized images
    cv2.imshow('Original Image', image)
    cv2.imshow('Equalized Image', equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
input_image_path = 'apple.jpeg'
output_image_path = 'path/to/your/output/equalized_image.jpg'

histogram_equalization(input_image_path, output_image_path)
