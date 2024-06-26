import cv2
import numpy as np
import matplotlib.pyplot as plt

def contrast_stretching(image):
    # Find the minimum and maximum pixel values
    min_val = np.min(image)
    max_val = np.max(image)
    
    # Apply contrast stretching
    stretched_image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    return stretched_image

def plot_histogram(image, title):
    # Calculate the histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    
    # Plot the histogram
    plt.figure()
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

# Example usage
input_image_path = 'forest.png'
output_image_path = 'path/to/your/output/contrast_stretched_image.jpg'

# Read the image in grayscale
image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Failed to read the image from {input_image_path}")
else:
    # Plot the histogram of the original image
    plot_histogram(image, "Histogram of Original Image")
    
    # Apply contrast stretching
    stretched_image = contrast_stretching(image)
    
    # Plot the histogram of the contrast-stretched image
    plot_histogram(stretched_image, "Histogram of Contrast-Stretched Image")
    
    # Save the contrast-stretched image
    cv2.imwrite(output_image_path, stretched_image)
    print(f"Contrast-stretched image saved as {output_image_path}")
    
    # Display the original and contrast-stretched images
    cv2.imshow('Original Image', image)
    cv2.imshow('Contrast-Stretched Image', stretched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()