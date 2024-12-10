import cv2
from ultralytics import YOLO

def classify_bolts_and_nuts_from_camera(model_path='C:/Users/Priyaa/boltsandnuts/best.pt', ref_width_pixels=100):
    """
    Classify bolts and nuts and estimate their actual size using a live camera feed,
    and save the annotated frame when the user presses 'c'.

    Args:
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
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Open the camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)  # Use the default camera (index 0)
    if not cap.isOpened():
        print("Error: Unable to access the camera. Ensure it is connected and not in use by another program.")
        return

    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'c' to capture and save the current frame. Press 'q' to quit the live feed.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read a frame from the camera. Exiting...")
            break

        # Perform inference
        results = model(frame)
        result = results[0]  # Get the first result

        # Annotate the frame with bounding boxes and actual sizes
        annotated_frame = result.plot()

        for detection in result.boxes.xyxy:  # Each detection includes [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, detection[:4])  # Bounding box coordinates
            width_pixels = x2 - x1
            height_pixels = y2 - y1

            # Convert dimensions to mm
            width_mm = width_pixels / pixels_per_mm
            height_mm = height_pixels / pixels_per_mm

            # Add size information to the frame
            label = f"{width_mm:.2f}mm x {height_mm:.2f}mm"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Show the live feed with annotations
        cv2.imshow("Classified Live Feed with Sizes", annotated_frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Quit the application
            break
        elif key == ord('c'):
            # Capture and save the current frame
            output_path = 'captured_frame_with_sizes.jpg'
            cv2.imwrite(output_path, annotated_frame)
            print(f"Captured frame saved as: {output_path}")

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the live camera detection
if __name__ == "__main__":
    classify_bolts_and_nuts_from_camera()