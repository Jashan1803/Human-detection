import cv2
import argparse
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov8.weights", "yolov8.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Function to detect humans in an image
def detect_humans(image_path, confidence_threshold):
    # Load image
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Process the outputs
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression to eliminate overlaps
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
    return [(boxes[i], confidences[i]) for i in range(len(boxes)) if i in indexes]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Human Detection with YOLO v8')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('confidence_threshold', type=float, help='Confidence threshold for detection')
    args = parser.parse_args()

    results = detect_humans(args.image_path, args.confidence_threshold)
    for box, confidence in results:
        print(f'Detected box: {box} with confidence: {confidence}')
