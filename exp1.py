import cv2

def read_and_write_image(input_path, output_path, is_color=True):
    # Read the image
    if is_color:
        image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was successfully read
    if image is None:
        print(f"Failed to read the image from {input_path}")
        return

    # Write the image to the output path
    success = cv2.imwrite(output_path, image)

    if success:
        print(f"Image successfully written to {output_path}")
    else:
        print(f"Failed to write the image to {output_path}")


input_image_path = 'E:\\Programs\\Python Codes\\DIP Lab\\apple.jpeg'
output_image_path_color = 'apple_gray.jpeg'



read_and_write_image(input_image_path, output_image_path_color, is_color=False)



