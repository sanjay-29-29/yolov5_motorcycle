import cv2
import torch
import pytesseract

model = torch.hub.load('', 'custom', path='yolov5l.pt', source='local') #change it according to file location

video_path = "video.mp4"

cap = cv2.VideoCapture(video_path)

class_to_detect = [2,3]


def apply_ocr(image, xmin, ymin, xmax, ymax):
    plate_roi = image[int(ymin):int(ymax), int(xmin):int(xmax)]
    gray_plate = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_plate, config='--psm 6')
    return text

if not cap.isOpened():
    print("Error: Unable to open the video file.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(frame_rgb)

    for detection in results.xyxy[0]:
        xmin, ymin, xmax, ymax, confidence, class_id = map(float, detection[:6])
        class_name = model.names[int(class_id)]
        if(class_id in class_to_detect and confidence>0.5):
            label = f"{class_name} {confidence:.2f}"
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            plate_text = apply_ocr(frame, xmin, ymin, xmax, ymax)
            label += f" | Plate: {plate_text}"
            cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("Confidence:", confidence)

            


    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


'''import cv2
import torch
import pytesseract

# Load YOLOv5 model
model = torch.hub.load('', 'custom', path='yolov5l.pt', source='local')  # Change it according to file location

# Path to the image file
image_path = "/home/pc/Documents/india-skoda-license-plate.jpg"

# Classes to detect
class_to_detect = [2, 3]

# Function to apply OCR to the license plate region
def apply_ocr(image, xmin, ymin, xmax, ymax):
    plate_roi = image[int(ymin):int(ymax), int(xmin):int(xmax)]
    gray_plate = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_plate, config='--psm 6')
    return text

# Read the image
frame = cv2.imread(image_path)

# Convert the image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Perform object detection
results = model(frame_rgb)

# Iterate through the detections
for detection in results.xyxy[0]:
    xmin, ymin, xmax, ymax, confidence, class_id = map(float, detection[:6])
    class_name = model.names[int(class_id)]
    if class_id in class_to_detect and confidence > 0.5:
        label = f"{class_name} {confidence:.2f}"
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        plate_text = apply_ocr(frame, xmin, ymin, xmax, ymax)
        label += f" | Plate: {plate_text}"
        cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("Confidence:", confidence)

# Display the processed image
cv2.imshow("Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
