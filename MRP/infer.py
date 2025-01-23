from ultralytics import YOLO

# Load the trained model
model = YOLO(r'path to trained yolo model best.onnx')

# Perform inference
results = model.predict(
    source=r'C:\Users\Saksh\ultralytics\yoloenv\OK_NOTOK\testdata',# Pass the test data folder path 
    save=True,
    imgsz=640
)

# Display results
for result in results:
    print(result.boxes)
