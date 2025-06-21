import cv2
from ultralytics import YOLO

MODEL_PATH = 'hotspot_runs\\train\\weights\\best.pt' 

# CAMERA_SOURCE = 0 
CAMERA_SOURCE = 'https://192.168.135.42:4343/video'
CONFIDENCE_THRESHOLD = 0.5 

CENTER_TOLERANCE_X = 0.05
CENTER_TOLERANCE_Y = 0.05

DISTANCE_CONSTANT = 50000
RELEASE_DISTANCE_THRESHOLD = 50 

try:
    model = YOLO(MODEL_PATH)
    print(f"YOLO model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Please ensure MODEL_PATH is correct and the model file exists.")
    exit()

cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video stream from {CAMERA_SOURCE}.")
    print("Please check camera connection or stream URL.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_center_x = frame_width // 2
frame_center_y = frame_height // 2

print(f"Camera opened: {frame_width}x{frame_height}")
print(f"Frame Center: ({frame_center_x}, {frame_center_y})")
print(f"Centering Tolerance X: {CENTER_TOLERANCE_X * frame_width:.0f} pixels")
print(f"Centering Tolerance Y: {CENTER_TOLERANCE_Y * frame_height:.0f} pixels")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # frame = cv2.flip(frame, 1)
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False) 

    command_text = "No Hotspot Detected"
    distance_text = "N/A"
    hotspot_detected = False
    target_box = None
    max_confidence = -1

    if results and results[0].boxes:
        for box in results[0].boxes:
            if box.conf.item() > max_confidence:
                max_confidence = box.conf.item()
                target_box = box

        if target_box is not None:
            hotspot_detected = True
            
            xyxy = target_box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            
            hotspot_center_x = (x1 + x2) // 2
            hotspot_center_y = (y1 + y2) // 2

            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (hotspot_center_x, hotspot_center_y), 5, (0, 0, 255), -1)

            if bbox_area > 0:
                estimated_distance = DISTANCE_CONSTANT / (bbox_area**0.5) 
                distance_text = f"Estimated Distance: {estimated_distance:.2f} units"
            else:
                estimated_distance = float('inf')
                distance_text = "Estimated Distance: Too Small"

            x_deviation = hotspot_center_x - frame_center_x
            y_deviation = hotspot_center_y - frame_center_y

            move_x = ""
            move_y = ""
            release_payload = False

            if abs(x_deviation) < CENTER_TOLERANCE_X * frame_width:
                move_x = "CENTER_X"
            elif x_deviation < 0:
                move_x = "MOVE_LEFT"
            else:
                move_x = "MOVE_RIGHT"

            if abs(y_deviation) < CENTER_TOLERANCE_Y * frame_height:
                move_y = "CENTER_Y"
            elif y_deviation < 0:
                move_y = "MOVE_UP"
            else:
                move_y = "MOVE_DOWN"

            command_text = f"X: {move_x}, Y: {move_y}"
            if move_x == "CENTER_X" and move_y == "CENTER_Y" and estimated_distance < RELEASE_DISTANCE_THRESHOLD:
                command_text += " -> RELEASE PAYLOAD!"
                release_payload = True
                cv2.putText(frame, "RELEASE PAYLOAD!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.circle(frame, (frame_center_x, frame_center_y), 5, (255, 0, 0), -1)
    cv2.putText(frame, f"Command: {command_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, distance_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Hotspot Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program exited.")