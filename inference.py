import cv2
import torch

model = torch.hub.load('', 'custom', path='yolov5l.pt', source='local') #change it according to file location

video_path = "video.mp4"

cap = cv2.VideoCapture(video_path)

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
        if(class_id==3 and confidence>0.5):
            label = f"{class_name} {confidence:.2f}"
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("Confidence:", confidence)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
