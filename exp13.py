import cv2
import numpy as np
import matplotlib.pyplot as plt

def roberts_cross_edge_detection(image_path, output_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Failed to read the image from {image_path}")
        return

    # Define the Roberts cross operator kernels
    kernel_x = np.array([[1, 0], [0, -1]], dtype=int)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=int)

    # Apply the kernels to the image
    roberts_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    roberts_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)

    # Compute the magnitude of the gradient
    magnitude = np.sqrt(roberts_x**2 + roberts_y**2)
    magnitude = np.uint8(np.absolute(magnitude))

    # Save the Roberts edge-detected image
    cv2.imwrite(output_path, magnitude)
    print(f"Roberts edge-detected image saved as {output_path}")

    # Plot the original and Roberts edge-detected images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(magnitude, cmap='gray')
    plt.title('Roberts Edge Detected Image')
    plt.axis('off')

    plt.show()

    # Display the images
    cv2.imshow('Original Image', image)
    cv2.imshow('Roberts Edge Detected Image', magnitude)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
input_image_path = 'forest.png'
output_image_path = 'roberts_edge_detected_image.jpg'

roberts_cross_edge_detection(input_image_path, output_image_path)
