import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(image, title):
    plt.figure()
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.hist(image.ravel(), bins=256, range=[0, 256], color='r')
    plt.xlim([0, 256])
    plt.show()

def calculate_cdf(hist):
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    return cdf_normalized

def histogram_matching(source, template):
    # Compute the histogram and cumulative distribution function (CDF) of the source image
    src_hist, bins = np.histogram(source.flatten(), 256, [0, 256])
    src_cdf = calculate_cdf(src_hist)

    # Compute the histogram and cumulative distribution function (CDF) of the template image
    tmpl_hist, bins = np.histogram(template.flatten(), 256, [0, 256])
    tmpl_cdf = calculate_cdf(tmpl_hist)

    # Create a lookup table to map pixel values from the source image to the template image
    lookup_table = np.zeros(256)
    tmpl_cdf_min = tmpl_cdf[tmpl_hist > 0].min()  # Avoid division by zero
    src_cdf_min = src_cdf[src_hist > 0].min()     # Avoid division by zero

    for src_pixel_val in range(256):
        src_val = src_cdf[src_pixel_val]
        closest_val = np.argmin(np.abs(tmpl_cdf - src_val))
        lookup_table[src_pixel_val] = closest_val

    # Map the source image through the lookup table
    matched = lookup_table[source]

    return matched

def histogram_matching_cv2(source_image_path, reference_image_path, output_image_path):
    # Read the source and reference images in grayscale
    source_image = cv2.imread(source_image_path, cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

    if source_image is None or reference_image is None:
        print(f"Failed to read the source or reference image")
        return

    # Perform histogram matching
    matched_image = histogram_matching(source_image, reference_image).astype(np.uint8)

    # Save the matched image
    cv2.imwrite(output_image_path, matched_image)
    print(f"Matched image saved as {output_image_path}")

    # Plot histograms
    plot_histogram(source_image, "Histogram of Source Image")
    plot_histogram(reference_image, "Histogram of Reference Image")
    plot_histogram(matched_image, "Histogram of Matched Image")

    # Display the images
    cv2.imshow('Source Image', source_image)
    cv2.imshow('Reference Image', reference_image)
    cv2.imshow('Matched Image', matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
source_image_path = 'apple.jpeg'
reference_image_path = 'forest.png'
output_image_path = 'path/to/your/output/matched_image.jpg'

histogram_matching_cv2(source_image_path, reference_image_path, output_image_path)
