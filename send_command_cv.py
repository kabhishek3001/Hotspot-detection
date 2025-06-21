import cv2
import numpy as np
import serial
import time

from send_command_advance import send_angles_to_arduino

ARDUINO_SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600
CONNECTION_TIMEOUT = 2

def is_contour_circle(contour: np.ndarray, min_circularity: float=0.8) -> bool:
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    
    if perimeter == 0:
        return False
    
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return circularity > min_circularity

def is_contour_centered(contour, frame_shape, tolerance=0.15):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return False
    
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    h, w = frame_shape[:2]
    center_x, center_y = w // 2, h // 2
    
    tol_x = int(w * tolerance)
    tol_y = int(h * tolerance)
    
    return (center_x - tol_x <= cX <= center_x + tol_x) and (center_y - tol_y <= cY <= center_y + tol_y)

def main():
    try:
        arduino = serial.Serial(ARDUINO_SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(CONNECTION_TIMEOUT)
    
    except serial.SerialException as e:
        print(f"Serial Error: {e}")
        return

    stream_url = "https://192.168.86.43:4343/video"

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Could not open Video.")
        arduino.close()
        return

    last_command_time = 0
    command_interval = 2  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            cv2.imshow("Contours", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        print(f"Largest contour area: {area}")

        display_frame = frame.copy()
        cv2.drawContours(display_frame, [largest_contour], -1, (0, 255, 0), 2)

        current_time = time.time()

        if is_contour_circle(largest_contour):
            print("Detected a circle.")

            if is_contour_centered(largest_contour, frame.shape):
                print("Circle is centered.")
                if current_time - last_command_time >= command_interval:
                    print("Sending '0' to Arduino.")
                    send_angles_to_arduino(arduino, 0, 0)
                    last_command_time = current_time
                
                else:
                    print("Command recently sent. Waiting before sending again.")
            else:
                print("Circle is not centered.")
                if current_time - last_command_time >= command_interval:
                    print("Sending '180' to Arduino.")
                    send_angles_to_arduino(arduino, 180, 180)
                    last_command_time = current_time
                
                else:
                    print("Command recently sent. Waiting before sending again.")
        else:
            print("Largest contour is not a circle.")

        cv2.imshow("Contours", display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    arduino.close()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()