from ultralytics import YOLO

# Load the trained model
model = YOLO(r'C:\Users\Saksh\ultralytics\yoloenv\OK_NOTOK\yolo_mrp_training\weights\best.pt')

# Perform inference
results = model.predict(
    source=r'C:\Users\Saksh\ultralytics\yoloenv\OK_NOTOK\testdata3',
    save=True,
    imgsz=640
)

# Display results
for result in results:
    print(result.boxes)