import serial
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import time
import logging
import subprocess

logging.basicConfig(level=logging.DEBUG)

#Used to kill port before attempting to connect
def kill_processes_using_port(port):
    try:
        # Find processes using the given port
        result = subprocess.check_output(['lsof', '-t', '-i:{}'.format(port)])

        # Kill each process
        for pid in result.decode('utf-8').split():
            subprocess.check_output(['kill', '-9', pid])

    except Exception as e:
        print(f"Error: {e}")
        print("No processes found using the port or error killing the process.")

class TurretCamGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('TurretCam GUI')
        
    

        # Camera index selection
        ttk.Label(root, text="Camera Index:").pack(pady=5)  # Added label for Camera Index
        self.camera_indices = self.get_available_camera_indices()
        self.camera_index = ttk.Combobox(root, values=self.camera_indices, width=5)
        if self.camera_indices:  # check if list is not empty
            self.camera_index.set(self.camera_indices[0])  # set default value to first index
        self.camera_index.pack(pady=20)

         # Label for Serial Connection Status
        ttk.Label(root, text="Serial Connection Status:").pack(pady=5)


        # Canvas for connection status light
        self.status_light = tk.Canvas(root, width=30, height=30, bg="white")
        self.status_light.pack(pady=20)
        self.status_circle = self.status_light.create_oval(5, 5, 25, 25, fill="red")  # Default to red (not connected)


        # Button to start video feed
        self.start_btn = ttk.Button(root, text="Start Video", command=self.start_video)
        self.start_btn.pack(pady=20)

        # Manual controls
        controls = ttk.Frame(root)
        controls.pack(pady=20)


        ttk.Button(controls, text='Up', command=lambda: self.send_command('U')).grid(row=0, column=1, pady=10)
        ttk.Button(controls, text='Down', command=lambda: self.send_command('D')).grid(row=2, column=1, pady=10)
        ttk.Button(controls, text='Left', command=lambda: self.send_command('L')).grid(row=1, column=0, padx=10)
        ttk.Button(controls, text='Right', command=lambda: self.send_command('R')).grid(row=1, column=2, padx=10)

        # Button to switch to the next target
        #self.next_target_btn = ttk.Button(root, text="Next Target", command=self.on_next_target_button_clicked)
        #self.next_target_btn.pack(pady=20)

        # Label to display the number of targets
        self.targets_label = ttk.Label(root, text="Targets Detected: 0")
        self.targets_label.pack(pady=10)

    #Used to control Arduino servo motors
    def send_command(self, command):
        #if not hasattr(self, 'ser') or not self.ser.is_open:
        #    print("Serial connection not open.")
        #    return
        if command not in ['C', 'R', 'L', 'U', 'D']:
            print(f"Unknown command: {command}")
            return
        try:
            self.ser.write(f"{command}\n".encode('utf-8'))
            print(f"Sent '{command}' command")
        except serial.SerialException as e:
            print(f"Failed to send '{command}' command due to {e}")

    #Start serial connection to Arduino and launch camera.py
    def start_video(self):
      
         # Check the serial connection
        try:
            self.ser = serial.Serial('/dev/ttyUSB0', 38400, timeout=1)
            self.ser.flush()
            self.status_light.itemconfig(self.status_circle, fill="green")
        except Exception as e:
            print(f"Failed to establish a serial connection: {e}")
            self.status_light.itemconfig(self.status_circle, fill="red")
            return

        cam_index = int(self.camera_index.get()) if self.camera_index.get() else None
        if cam_index is None:
            print("Please select a camera index.")
            return
      
        # Start camera.py in a separate subprocess
        try:
            self.camera_subprocess = subprocess.Popen(['python', 'camera.py', '--camindex', str(cam_index)])
            print("camera.py started successfully.")
        except Exception as e:
            print(f"Error starting camera.py: {e}")

    #Find total number of camera indexs available
    def get_available_camera_indices(self):
        max_tested = 10
        available_indices = []
        for index in range(max_tested):
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                available_indices.append(index)
                cap.release()
            else:
                break
        return available_indices

             
    #Clean up function         
    def on_closing(self):
        # Close serial connection if it's open
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()

        # Terminate the camera subprocess if it exists
        if hasattr(self, 'camera_subprocess'):
            self.camera_subprocess.terminate()

        self.root.destroy()




root = tk.Tk()
root.geometry("380x480")
app = TurretCamGUI(root)
root.protocol("WM_DELETE_WINDOW", app.on_closing)  # Moved the protocol setup outside the class.
root.mainloop()
