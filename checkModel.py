from ultralytics import YOLO

# Load a pretrained YOLOv10n model
model = YOLO("yolov9c.pt")

# Perform object detection on an image
results = model("img.jpg")

# Display the results
results[0].show()