import multiprocessing
import logging
import socket
import subprocess
import os
from queue import Queue
from threading import Thread, Event
import errno

logging.basicConfig(level=logging.DEBUG)

class CommunicationServer:
    def __init__(self, port_dict):
        self.port_dict = port_dict
        self.connections = {}
        self.events = {}
        self.queues = {}
        self.threads = {}
        self.server_sockets = {}
        self.gui_subprocess = None

    def kill_processes_on_ports(self):
        for module, port in self.port_dict.items():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.connect(('localhost', port))
                    logging.info(f"Port {port} is occupied, attempting to free it.")
                    if os.name == 'posix':  # Unix/Linux/MacOS/BSD/etc
                        os.system(f"lsof -t -i:{port} | xargs kill -9")
                    elif os.name == 'nt':  # Windows
                        os.system(f"FOR /F \"usebackq tokens=5\" %p IN (`netstat -ano ^| findstr :{port}`) DO taskkill /F /PID %p")
                except socket.error as e:
                    if e.errno == errno.ECONNREFUSED:
                        logging.info(f"Port {port} is free: {e}")
                    else:
                        logging.error(f"Error checking port {port}: {e}")

    def start_gui_subprocess(self):
        try:
            self.gui_subprocess = subprocess.Popen(['python', 'GUI.py'])
            logging.info("GUI.py started successfully.")
        except Exception as e:
            logging.error(f"Error starting GUI.py: {e}")

    def wait_for_connection(self, module_name):
        port = self.port_dict.get(module_name.upper())
        self.kill_processes_on_ports()
        self.start_gui_subprocess()

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('localhost', port))
        server_socket.listen(1)
        self.server_sockets[module_name] = server_socket

        logging.info(f"Waiting for connection on port {port}")
        conn, addr = server_socket.accept()
        logging.info(f"Connected by {addr}")

        # Handshake
        conn.sendall(b'hello')
        data = conn.recv(1024)
        if data.decode('utf-8') == 'hello':
            logging.info("Handshake successful")
            self.events[module_name].set()
            self.queues[module_name].put(conn)
        else:
            logging.info("Handshake failed")

    def listen_for_messages(self, module_name):
        self.events[module_name].wait()
        conn = self.queues[module_name].get()
        while True:
            data = conn.recv(1024)
            if data:
                logging.info(f"Received: {data.decode('utf-8')}")

    def start(self):
        for module_name in self.port_dict.keys():
            self.events[module_name] = Event()
            self.queues[module_name] = Queue()
            connection_thread = Thread(target=self.wait_for_connection, args=(module_name,))
            listener_thread = Thread(target=self.listen_for_messages, args=(module_name,))
            connection_thread.start()
            listener_thread.start()
            self.threads[module_name] = (connection_thread, listener_thread)

    def stop(self):
        for server_socket in self.server_sockets.values():
            server_socket.close()
        for conn in self.connections.values():
            conn.close()
        for connection_thread, listener_thread in self.threads.values():
            connection_thread.join()
            listener_thread.join()
        if self.gui_subprocess:
            self.gui_subprocess.terminate()
            self.gui_subprocess.wait()

if __name__ == '__main__':
    port_dict = {
        'GUI': 6000,
        'CAMERA': 6001,
        'SERVODRIVES': 6002,
        #'MAIN': 6003
    }
    comm_server = CommunicationServer(port_dict)
    comm_server.start()

    # The server will now be listening for connections and messages.
    # You can add your application logic here, and use comm_server to send/receive messages.

    # When shutting down:
    comm_server.stop()
