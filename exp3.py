import cv2
import numpy as np

def intensity_level_slicing(image_path, lower_bound, upper_bound, high_intensity=255, low_intensity=0):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Failed to read the image from {image_path}")
        return None

    # Create an output image with the same dimensions
    sliced_image = np.zeros_like(image)

    # Apply intensity level slicing
    sliced_image[(image >= lower_bound) & (image <= upper_bound)] = high_intensity
    sliced_image[(image < lower_bound) | (image > upper_bound)] = low_intensity

    return sliced_image

# Example usage
input_image_path = 'apple.jpeg'
output_image_path = 'sliced_image.jpg'
lower_bound = 100
upper_bound = 200

sliced_image = intensity_level_slicing(input_image_path, lower_bound, upper_bound)

if sliced_image is not None:
    # Save the output image
    cv2.imwrite(output_image_path, sliced_image)
    print(f"Intensity level sliced image saved as {output_image_path}")

    # Display the input and output images
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Original Image', input_image)
    cv2.imshow('Intensity Level Sliced Image', sliced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
