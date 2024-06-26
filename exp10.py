import cv2
import numpy as np
import matplotlib.pyplot as plt

def power_law_transformation(image_path, output_path, gamma):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Failed to read the image from {image_path}")
        return

    # Normalize the image to the range [0, 1]
    normalized_image = image / 255.0

    # Apply power-law transformation
    c = 1.0  # Scaling constant
    transformed_image = c * (normalized_image ** gamma)

    # Scale back to the range [0, 255] and convert to uint8
    transformed_image = np.uint8(transformed_image * 255)

    # Save the transformed image
    cv2.imwrite(output_path, transformed_image)
    print(f"Power-law transformed image saved as {output_path}")

    # Plot the original and power-law transformed images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(transformed_image, cmap='gray')
    plt.title(f'Power-law Transformed Image (gamma={gamma})')
    plt.axis('off')

    plt.show()

    # Display the images
    cv2.imshow('Original Image', image)
    cv2.imshow('Power-law Transformed Image', transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
input_image_path = 'apple.jpeg'
output_image_path = 'power_law_transformed_image.jpg'
gamma_value = 2.0  # Example gamma value

power_law_transformation(input_image_path, output_image_path, gamma_value)
