import cv2
import torch
import torchvision.models as models
from helmet_model import binary_classifier
import numpy as np
from flask import Flask, Response

app = Flask(__name__)

model = torch.hub.load('', 'custom', path='best.pt', source='local')

helmet_model =binary_classifier.BinaryClassifier()

helmet_model.load_state_dict(torch.load('helmet_model.pth'))

video_path = "VID_20240213_175219.mp4"

def stream():

    cap = cv2.VideoCapture(video_path)


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

            #if xmin_roi <= xmin <= xmax_roi and ymin_roi <= ymin <= ymax_roi:
            label = f"{class_name} {confidence:.2f}"
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

            if(class_name == 'motorcycle'):
                cropped_img = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
                cropped_img = cv2.resize(cropped_img, (64, 64))  
                cropped_img = torch.from_numpy(cropped_img).permute(2, 0, 1).unsqueeze(0).float() 

                with torch.no_grad():
                    output = helmet_model(cropped_img)
                    helmet_present = output.item() > 0.5

                    if helmet_present:
                        helmet_text = "Helmet: False"
                    else:
                        helmet_text = "Helmet: True"

                
                cv2.putText(frame, helmet_text, (int(xmin), int(ymin) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print("Confidence:", confidence)
            else:
                label += f""
                cv2.putText(frame, label,(int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print("Confidence:", confidence)
            
        overlay = frame.copy()
        cv2.rectangle(overlay, (xmin_roi, ymin_roi), (xmax_roi, ymax_roi), (0, 255, 0), -1)  
        alpha = 0.3  
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/webcam')
def webcam_display():
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
