import cv2
import numpy as np
import matplotlib.pyplot as plt
def sobel_edge_detection(image_path, output_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Failed to read the image from {image_path}")
        return

    # Apply Sobel operator in X direction
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    
    # Apply Sobel operator in Y direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the magnitude of the gradient
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = np.uint8(np.absolute(sobel_magnitude))

    # Save the Sobel edge-detected image
    cv2.imwrite(output_path, sobel_magnitude)
    print(f"Sobel edge-detected image saved as {output_path}")

    # Plot the original and Sobel edge-detected images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sobel_magnitude, cmap='gray')
    plt.title('Sobel Edge Detected Image')
    plt.axis('off')

    plt.show()

    # Display the images
    cv2.imshow('Original Image', image)
    cv2.imshow('Sobel Edge Detected Image', sobel_magnitude)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
input_image_path = 'apple.jpeg'
output_image_path = 'sobel_edge_detected_image.jpg'

sobel_edge_detection(input_image_path, output_image_path)
