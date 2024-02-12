import cv2
import torch
import easyocr
from tensorflow.lite.python.interpreter import Interpreter
import numpy as np

reader = easyocr.Reader(['en'])

model = torch.hub.load('', 'custom', path='yolov5l.pt', source='local') #change it according to file location

video_path = "demo.mp4"

cap = cv2.VideoCapture(video_path)

class_to_detect = [2,3]


tflite_model_path = 'detect.tflite'
interpreter = Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]


def apply_ocr(image, xmin, ymin, xmax, ymax):
    plate_roi = image[int(ymin):int(ymax), int(xmin):int(xmax)]
    gray_plate = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_plate, config='--psm 6')
    return text

def tflite(image, min_conf, input_details, output_details, imH, imW):
    input_data = (np.float32(image) - 127.5) / 127.5
    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] 
    classes = interpreter.get_tensor(output_details[3]['index'])[0] 
    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    
    detected_text = []  
    
    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            cropped_img = image[ymin:ymax, xmin:xmax]  
            
            ocr_result = reader.readtext(cropped_img)
            
            for result in ocr_result:
                text = result[1]
                detected_text.append(text) 
    
    single_text = ' '.join(detected_text)
    return single_text



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
            cropped_img = cv2.resize(frame_rgb[int(ymin):int(ymax), int(xmin):int(xmax)], (width, height))

            plate_text = tflite(cropped_img, 0.5, input_details, output_details, cropped_img.shape[0], cropped_img.shape[1])
            label += f" | Plate: {plate_text}"

            cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("Confidence:", confidence)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

'''
import cv2
import torch
import easyocr
from tensorflow.lite.python.interpreter import Interpreter
import numpy as np

reader = easyocr.Reader(['en'])

model = torch.hub.load('', 'custom', path='yolov5l.pt', source='local') #change it according to file location

image_path = "/home/pc/folder/images/Cars77.png"

class_to_detect = [2,3]

tflite_model_path = 'detect.tflite'
interpreter = Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

def tflite(image, min_conf, input_details, output_details, imH, imW):
    input_data = (np.float32(image) - 127.5) / 127.5
    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] 
    classes = interpreter.get_tensor(output_details[3]['index'])[0] 
    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    
    detected_text = []  # List to store detected texts from all detected regions
    
    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            # No need to draw rectangles and labels here
            # Perform OCR on the detected region
            cropped_img = image[ymin:ymax, xmin:xmax]  # Crop the detected region
            
            # Apply OCR using EasyOCR
            ocr_result = reader.readtext(cropped_img)
            
            # Filter OCR results
            for result in ocr_result:
                text = result[1]
                detected_text.append(text)  # Append detected text to the list
    
    # Concatenate all detected texts into a single string
    single_text = ' '.join(detected_text)
    print(single_text)
    return single_text

# Read the image
frame = cv2.imread(image_path)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

results = model(frame_rgb)

for detection in results.xyxy[0]:

    xmin, ymin, xmax, ymax, confidence, class_id = map(float, detection[:6])
    class_name = model.names[int(class_id)]

    if(class_id in class_to_detect and confidence > 0.5):

        label = f"{class_name} {confidence:.2f}"
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cropped_img = cv2.resize(frame_rgb[int(ymin):int(ymax), int(xmin):int(xmax)], (width, height))

        plate_text = tflite(cropped_img, 0.5, input_details, output_details, cropped_img.shape[0], cropped_img.shape[1])
        label += f" | Plate: {plate_text}"

        cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("Confidence:", confidence)

cv2.imshow("Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
