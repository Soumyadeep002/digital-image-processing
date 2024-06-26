import cv2
import numpy as np
import matplotlib.pyplot as plt

def log_transformation(image_path, output_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Failed to read the image from {image_path}")
        return

    # Apply log transformation
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(1 + image))

    # Convert the image to uint8 type
    log_image = np.array(log_image, dtype=np.uint8)

    # Save the transformed image
    cv2.imwrite(output_path, log_image)
    print(f"Log transformed image saved as {output_path}")

    # Plot the original and log transformed images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(log_image, cmap='gray')
    plt.title('Log Transformed Image')
    plt.axis('off')

    plt.show()

    # Display the images
    cv2.imshow('Original Image', image)
    cv2.imshow('Log Transformed Image', log_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
input_image_path = 'forest.png'
output_image_path = 'log_transformed_image.jpg'

log_transformation(input_image_path, output_image_path)
