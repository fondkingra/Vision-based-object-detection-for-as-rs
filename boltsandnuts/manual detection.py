import cv2
from ultralytics import YOLO
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import random

def classify_bolts_and_nuts_with_reference(image_path, model_path='C:/Users/Priyaa/boltsandnuts/best.pt', ref_width_pixels=100):
    """
    Classify bolts and nuts and estimate their actual size using an M5 bolt as a reference.

    Args:
        image_path (str): Path to the input image for classification.
        model_path (str): Path to the YOLO model.
        ref_width_pixels (int): Width in pixels of the reference M5 bolt.
    """
    # Known dimensions of the M5 bolt (in mm)
    reference_real_size_mm = 9.2  # Across corners (B)

    # Calculate pixels per mm using the hard-coded reference width
    pixels_per_mm = ref_width_pixels / reference_real_size_mm
    print(f"Pixels per mm: {pixels_per_mm:.2f}")

    # Load the pre-trained YOLO model
    print("Loading YOLO model...")
    model = YOLO(model_path)

    # Load and resize the input image
    print("Loading and resizing input image...")
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return
    resized_image = cv2.resize(image, (640, 640))  # Resize for faster processing

    # Perform inference
    print("Performing inference...")
    results = model(resized_image)
    result = results[0]  # Get the first result

    # Annotate the image with bounding boxes and actual sizes
    annotated_image = resized_image.copy()

    for detection in result.boxes.xyxy:  # Each detection includes [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, detection[:4])  # Bounding box coordinates
        width_pixels = x2 - x1
        height_pixels = y2 - y1

        # Convert dimensions to mm
        width_mm = width_pixels / pixels_per_mm
        height_mm = height_pixels / pixels_per_mm

        # Generate a random color for each box
        box_color = [random.randint(0, 255) for _ in range(3)]

        # Add size information in bold text to the image
        label = f"{width_mm:.2f}mm x {height_mm:.2f}mm"
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), box_color, 2)  # Draw rectangle
        cv2.putText(
            annotated_image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,  # Font size
            box_color,
            2,  # Text thickness
            cv2.LINE_AA,
        )

    # Show the output image with annotations
    cv2.imshow("Classified Image with Sizes", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the output image
    output_path = 'classified_output_with_sizes.jpg'
    cv2.imwrite(output_path, annotated_image)
    print(f"Output image saved at: {output_path}")

# Main function to process the images
def main():
    # Create a Tkinter root window and hide it
    Tk().withdraw()

    # Open file dialog to select the input image
    image_path = askopenfilename(title="Select Image for Classification", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if not image_path:
        print("No image selected. Exiting...")
        return

    # Call the function with the selected image
    classify_bolts_and_nuts_with_reference(image_path)

# Run the script
if __name__ == "__main__":
    main()
