from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-seg.pt")  # load a pretrained segmentation model (recommended for training)

# Train the model on the Package Segmentation dataset
results = model.train(data="package-seg.yaml", epochs=100, imgsz=640)

# Validate the model
results = model.val()

# Perform inference on an image
results = model('rgb.png', conf=0.15)

results[0].show()