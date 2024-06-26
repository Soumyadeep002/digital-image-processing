import cv2
import numpy as np

def bit_plane_slicing(image_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Failed to read the image from {image_path}")
        return

    # Get the dimensions of the image
    rows, cols = image.shape

    # Create a list to hold each bit plane image
    bit_planes = []

    # Extract each bit plane
    for i in range(8):
        # Create an empty image to hold the bit plane
        bit_plane = np.zeros((rows, cols), dtype=np.uint8)
        
        # Extract the bit plane
        bit_plane[:, :] = (image[:, :] & (1 << i)) >> i
        
        # Multiply by 255 to visualize the bit plane (optional)
        bit_plane *= 255

        # Add the bit plane to the list
        bit_planes.append(bit_plane)

        # Save the bit plane image (optional)
        output_path = f'bit_plane_{i}.png'
        cv2.imwrite(output_path, bit_plane)
        print(f"Bit plane {i} saved as {output_path}")

    return bit_planes

# Example usage
input_image_path = 'apple.jpeg'
bit_planes = bit_plane_slicing(input_image_path)

# Display the bit planes
for i, bit_plane in enumerate(bit_planes):
    cv2.imshow(f'Bit Plane {i}', bit_plane)

cv2.waitKey(0)
cv2.destroyAllWindows()
