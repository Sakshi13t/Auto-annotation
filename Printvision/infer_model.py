import os
import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO(r'C:\Users\xyz\ultralytics\yoloenv\printvision\runs\train\custom_yolov8n10\weights\best.pt')

# Specify the folder containing test images
test_image_folder = r"C:\Users\xyz\ultralytics\yoloenv\printvision\pvdata\test\images"
output_folder = r'C:\Users\xyz\ultralytics\yoloenv\printvision\output_images'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Parameters for NMS and confidence
conf_threshold = 0.4  # Confidence threshold (adjust as needed)
iou_threshold = 0.5   # IoU threshold for NMS (adjust as needed)

# Iterate over all images in the test folder
for image_name in os.listdir(test_image_folder):
    image_path = os.path.join(test_image_folder, image_name)

    # Check if the file is an image
    if image_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
        # Perform inference on the image
        results = model.predict(
            source=image_path,
            imgsz=640,
            conf=conf_threshold,  # Set the confidence threshold
            iou=iou_threshold,    # Set the IoU threshold for NMS
            save=False
        )

        # Iterate over the results for each image
        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers

                # Draw bounding box on the image
                cv2.rectangle(result.orig_img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green for bounding box

                # Get class label
                class_name = model.names[int(cls)]
                
                # Annotate class name (without confidence score)
                font_size = 5.0
                font_thickness = 9
                cv2.putText(result.orig_img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), font_thickness)

            # Save the processed image
            output_image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_image_path, result.orig_img)
            print(f"Processed image saved at {output_image_path}")

print("All images processed successfully!")
