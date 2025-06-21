import serial
import time

ARDUINO_SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600
CONNECTION_TIMEOUT = 2

def send_angles_to_arduino(ser: serial.Serial, angle1: int, angle2: int) -> bool:
    """
        Sends two angles to the Arduino.
    
        Args:
            ser (serial.Serial): The serial connection object.
            angle1 (int): The angle for the first servo (0-180).
            angle2 (int): The angle for the second servo (0-180).
        
        Returns:
            bool: True if command sent successfully, False otherwise.
    """
    
    try:
        if not (0 <= angle1 <= 180 and 0 <= angle2 <= 180):
            print("Error: Angles must be between 0 and 180 degrees.")
            return False

        # Format the command string (e.g., "90,45\n")
        command = f"{angle1},{angle2}\n"

        ser.write(command.encode('utf-8'))
        print(f"Sent command: {command.strip()}")
        
        return True
    
    except serial.SerialException as e:
        print(f"Error sending data: {e}")
        return False
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    arduino = None
    
    try:
        print(f"Attempting to connect to Arduino on {ARDUINO_SERIAL_PORT} at {BAUD_RATE} baud...")
        arduino = serial.Serial(ARDUINO_SERIAL_PORT, BAUD_RATE, timeout=1)
        
        time.sleep(CONNECTION_TIMEOUT)
        print("Successfully connected to Arduino.")

        if arduino.in_waiting > 0:
            initial_message = arduino.readline().decode('utf-8').strip()
            
            print(f"Arduino says: {initial_message}")
            while arduino.in_waiting > 0:
                 print(f"Arduino says: {arduino.readline().decode('utf-8').strip()}")

        # send_angles_to_arduino(arduino, 90, 45)
        # time.sleep(1)

        while True:
            try:
                print("\nEnter servo angles (0-180) or type 'exit' to quit.")
                input_str = input("Enter angle1,angle2 (e.g., 30,150): ")

                if input_str.lower() == 'exit':
                    break

                parts = input_str.split(',')
                if len(parts) == 2:
                    angle_servo1 = int(parts[0].strip())
                    angle_servo2 = int(parts[1].strip())
                    send_angles_to_arduino(arduino, angle_servo1, angle_servo2)

                    time.sleep(0.1)
                    if arduino.in_waiting > 0:
                        response = arduino.readline().decode('utf-8').strip()
                        print(f"Arduino response: {response}")
                else:
                    print("Invalid input format. Please use: angle1,angle2")

            except ValueError:
                print("Invalid input. Please enter numbers for angles.")
            except KeyboardInterrupt:
                print("\nExiting...")
                break

    except serial.SerialException as e:
        print(f"Serial Error: {e}")
        print("Failed to connect to Arduino. Please check:")
        print("1. Is the Arduino plugged in and powered on?")
        print(f"2. Is '{ARDUINO_SERIAL_PORT}' the correct port? (See instructions below)")
        print("3. Is the Arduino IDE's Serial Monitor closed? (It can block the port)")
        print("4. Do you have the necessary permissions to access the serial port?")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    finally:
        if arduino and arduino.is_open:
            arduino.close()
            print("Serial connection closed.")
