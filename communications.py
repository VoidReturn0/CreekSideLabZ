import logging

logging.basicConfig(level=logging.DEBUG)

# Define the port numbers
GUI_PORT = 6000
CAMERA_PORT = 6001

    """Read a message from a connection."""
    data = ""
    while True:
        chunk = conn.recv().decode('utf-8')
        data += chunk
        if "<EOM>" in data:
            return data.replace("<EOM>", "")