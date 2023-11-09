import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import logging
import threading
import tkinter.messagebox as messagebox  # Import messagebox for popup

# Assuming CommunicationServer is a class in comms.py that handles all IPC and subprocess management
from comms import CommunicationServer

logging.basicConfig(level=logging.DEBUG)
logging.info("GUI_started")

class TurretCamGUI:
    def __init__(self, root, comm_instance):
        logging.info("GUI_init started")
        self.comm = comm_instance  # Use the provided Communications instance
        self.root = root  # Store the Tk root object for later use
        
        # Camera index selection
        ttk.Label(root, text="Camera Index:").pack(pady=5)
        self.camera_indices = self.get_available_camera_indices()
        self.camera_index = ttk.Combobox(root, values=self.camera_indices, width=5)
        if self.camera_indices:
            self.camera_index.set(self.camera_indices[0])
        self.camera_index.pack(pady=20)

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

        # Label to display the number of targets
        self.targets_label = ttk.Label(root, text="Targets Detected: 0")
        self.targets_label.pack(pady=10)

        # Start the listening thread
        self.start_listening_thread()
        
        logging.info("GUI_init finished")

    # Method to handle incoming messages
    def handle_message(self, message):
        logging.info(f"Received message: {message}")
        # Here you can add logic to handle different types of messages
        if message == "some_command":
            # Do something
            pass

    # Add a new method to listen for incoming messages
    def listen_for_messages(self):
        while True:
            message = self.comm.receive_message()
            if message:
                self.handle_message(message)

    # Add a new method to start the listening thread
    def start_listening_thread(self):
        self.listening_thread = threading.Thread(target=self.listen_for_messages)
        self.listening_thread.daemon = True
        self.listening_thread.start()
        logging.debug("GUI_listening thread started")

    # Used to control Arduino servo motors
    def send_command(self, command):
        logging.info("GUI_send command")
        self.comm.send_command(command)

    # Start serial connection to Arduino and launch camera.py
    def start_video(self):
        logging.info("GUI_start_video")
        cam_index = int(self.camera_index.get()) if self.camera_index.get() else None
        if cam_index is not None:
            self.comm.start_video(cam_index)
        else:
            messagebox.showerror("Error", "Please select a camera index.")

    # Find total number of camera indices available
    def get_available_camera_indices(self):
        logging.info("GUI_get_available_camera_indices")
        return self.comm.get_available_camera_indices()

    # Clean up function
    def on_closing(self):
        logging.info("GUI_on_closing: Starting cleanup")
        self.comm.cleanup()
        logging.info("GUI_on_closing: Cleanup complete")
        self.root.destroy()

if __name__ == "__main__":
    logging.info("GUI_main started")
    
    # Initialize CommunicationServer instance
    comm = CommunicationServer()

    root = tk.Tk()
    root.geometry("380x480")
    app = TurretCamGUI(root, comm)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
