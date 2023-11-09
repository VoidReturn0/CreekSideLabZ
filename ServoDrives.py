import serial
import logging
from threading import Thread
from comms import Communications
import time

logging.basicConfig(level=logging.DEBUG)

# At the top of your script
X_TOLERANCE = 15
Y_TOLERANCE = 15
X_HYSTERESIS = 55
Y_HYSTERESIS = 55

class ServoDrives:
    def __init__(self):
        logging.debug("ServoDrives: __init__ started")
        # Initialize Communications with message handler
        self.comm = Communications('ServoDrives', message_handler=self.handle_incoming_message)
  
        # Initialize Serial Connection to Arduino
        self.init_serial_connection()
        
    
        # Initialize tolerance and hysteresis values
        self.x_tolerance = X_TOLERANCE
        self.y_tolerance = Y_TOLERANCE
        self.x_hysteresis = X_HYSTERESIS
        self.y_hysteresis = Y_HYSTERESIS

         # Initialize last command time
        self.last_command_time = time.time()

        logging.debug("ServoDrives: __init__ finished")
        


    #intitalize serial connections
    def init_serial_connection(self):
        try:
            self.ser = serial.Serial('/dev/ttyUSB0', 38400, timeout=1)
            self.ser.flush()
            logging.info("Serial connection established.")
        except Exception as e:
            logging.error(f"Failed to establish a serial connection: {e}")

    

    #listens for messages from camera.py
    def listen_for_messages(self):
        logging.debug("Servo_listening")
        try:
            logging.debug("ServoDrives: listen_for_messages started")
            while True:
                logging.debug("Inside listen_for_messages loop")  # <-- Add this line
                message = self.comm.receive_message()
                if message:
                    logging.debug(f"ServoDrives: Received message: {message}")
                    self.handle_incoming_message(message)
        except Exception as e:
            logging.error(f"Exception in listen_for_messages: {e}")

    #Process incoming offests from camera.py
    def handle_incoming_message(self, message):
        logging.debug(f"ServoDrives: handle_incoming_message started with message: {message}")
        x_offset, y_offset = map(float, message.split(','))  # Assuming message is "x_offset,y_offset"
        self.handle_offsets(x_offset, y_offset)
        logging.debug("ServoDrives: handle_incoming_message finished")

    #Used to control Arduino servo motors
    #def send_command(self, command):
     #   logging.info("GUI_send command def")

     #   if command not in ['C', 'R', 'L', 'U', 'D']:
     #       print(f"Unknown command: {command}")
     #       return
     #   try:
     #       self.ser.write(f"{command}\n".encode('utf-8'))
     #       print(f"Sent '{command}' command")
     #   except serial.SerialException as e:
     #      print(f"Failed to send '{command}' command due to {e}")

    #Used to process offsets and send arduino commands
    def handle_offsets(self, x_offset, y_offset):
        logging.debug(f"ServoDrives: handle_offsets started with x_offset: {x_offset}, y_offset: {y_offset}")
        current_time = time.time()
        if current_time - self.last_command_time >= 0.10:
            if abs(x_offset) > self.y_tolerance + self.y_hysteresis:
                if x_offset > 0:
                    self.send_command('D')
                else:
                    self.send_command('U')

            if abs(y_offset) > self.x_tolerance + self.x_hysteresis:
                if y_offset > 0:
                    self.send_command('L')
                else:
                    self.send_command('R')

            self.last_command_time = current_time
        logging.debug("ServoDrives: handle_offsets finished")



if __name__ == "__main__":
    
    logging.debug("ServoDrives: Main execution started")
    servo_drives = ServoDrives()
    logging.debug("ServoDrives: Main execution finished")
    
