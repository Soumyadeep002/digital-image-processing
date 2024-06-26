import cv2
import numpy as np

def generate_negative_image(image_path, output_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Failed to read the image from {image_path}")
        return

    # Generate the negative image
    negative_image = 255 - image

    # Save the negative image
    cv2.imwrite(output_path, negative_image)
    print(f"Negative image saved as {output_path}")

    # Display the original and negative images
    cv2.imshow('Original Image', image)
    cv2.imshow('Negative Image', negative_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
input_image_path = 'apple_gray.jpeg'
output_image_path = 'negative_image.jpg'

generate_negative_image(input_image_path, output_image_path)