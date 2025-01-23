from ultralytics import YOLO

# Load YOLOv8n model 
model = YOLO('yolov8n.pt')

# Train the model
model.train(
    data=r'C:\Users\xyz\ultralytics\printvis\dataset.yaml', 
    epochs=50, 
    augment=True,                  
    imgsz=640,                   
    batch=16,                    # Batch size
    project='runs/train',        # Save results to runs/train
    name='custom_yolov8n',       # Model name
    save_period = 1,
    device = 'cpu'
)

# Validate the model on the validation set
metrics = model.val(data='dataset.yaml')
print(metrics)

#exporting the model
model.export(format='onnx') 


