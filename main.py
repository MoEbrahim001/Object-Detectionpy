import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (best.pt is your trained model)
model = YOLO('best.pt')
# Replace with the correct path to your 'best.pt' file

# Load the COCO class names from coco.txt file
def load_coco_classes(file_path):
    with open(file_path, 'r') as f:
        class_names = f.read().splitlines()  # Read each line as a class name
    return class_names

# Path to your coco.txt file (update the path if needed)
coco_classes = load_coco_classes('coco.txt')  # Replace with your coco.txt file path

# Read the image file (Pen.jpeg)
img = cv2.imread('PEN88.jpeg')  # Replace with the correct path to your image file

# Resize the image to make it smaller (for example, resizing to 50% of original size)
scale_percent = 50  # Percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize the image
resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# Run YOLOv8 inference on the resized image
results = model(resized_img)

# Loop through the detections and draw bounding boxes with labels
for result in results:
    boxes = result.boxes  # Get bounding boxes
    for box in boxes:
        # Get box coordinates and confidence score
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        conf = box.conf[0]  # Confidence score

        # Get the class ID and class name
        class_id = int(box.cls[0])
        class_name = coco_classes[class_id]  # Look up class name from coco.txt

        # Draw the bounding box and label on the resized image
        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(resized_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

# Display the resized image with detected objects
cv2.imshow('Detected Objects', resized_img)

# Wait for a key press and close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()

