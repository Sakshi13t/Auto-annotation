from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO(r"C:\Users\Saksh\ultralytics\yoloenv\OK_NOTOK\yolo_mrp_training\weights\last.pt")  

# Train the model
model.train(
    data=r"C:\Users\Saksh\ultralytics\yoloenv\OK_NOTOK\data.yaml",  # Path to data.yaml
    epochs=50,                 
    augment = True,
    imgsz=640,                 # Image size
    batch=16,                  # Batch size
    name="yolo_mrp_training",
    resume = True,
    save_period = 1,
    device="cpu",
    project=r"C:\Users\Saksh\ultralytics\yoloenv\OK_NOTOK"                 
)

# Evaluate the model on test data 
metrics = model.val()
print(metrics)

model.export(format="onnx")  # Export to ONNX format
