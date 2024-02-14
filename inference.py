import cv2
import torch
import easyocr
from tflite_runtime.interpreter import Interpreter
import numpy as np

reader = easyocr.Reader(['en'])

model = torch.hub.load('', 'custom', path='last.pt', source='local') #change it according to file location

video_path = "video.mp4"

cap = cv2.VideoCapture(video_path)

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
    
    detected_text = []  
    
    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            cropped_img = image[ymin:ymax, xmin:xmax]  
            gray_plate = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            ocr_result = reader.readtext(gray_plate)
                
            for result in ocr_result:
                text = result[1]
                detected_text.append(text) 
    
    single_text = ' '.join(detected_text)
    return single_text


if not cap.isOpened():
    print("Error: Unable to open the video file.")
    exit()


xmin_roi, ymin_roi, xmax_roi, ymax_roi = 43, 347, 1245, 704 

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4)))) #(saving the video as .mp4)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(frame_rgb)

    for detection in results.xyxy[0]:

        xmin, ymin, xmax, ymax, confidence, class_id = map(float, detection[:6])
        class_name = model.names[int(class_id)]

        if xmin_roi <= xmin <= xmax_roi and ymin_roi <= ymin <= ymax_roi:
            label = f"{class_name} {confidence:.2f}"
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

            if(class_name == 'motorcycle'):
                cropped_img = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
                cv2.imwrite("motorcycle_frame.jpg", cropped_img) #(saving motorcycle class)

            cropped_img = cv2.resize(frame_rgb[int(ymin):int(ymax), int(xmin):int(xmax)], (width, height))
            label += f" | Plate:"
            cv2.putText(frame, label,(int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("Confidence:", confidence)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (xmin_roi, ymin_roi), (xmax_roi, ymax_roi), (0, 255, 0), -1)  
    alpha = 0.3  
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    out.write(frame)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
