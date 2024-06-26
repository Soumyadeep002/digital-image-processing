import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(image_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Failed to read the image from {image_path}")
        return

    # Calculate the histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # Plot the histogram
    plt.figure()
    plt.title("Histogram of Grayscale Image")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

# Example usage
input_image_path = 'apple.jpeg'
plot_histogram(input_image_path)
