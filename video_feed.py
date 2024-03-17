import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from flask import Flask, Response
import threading
import os
from std_msgs.msg import Int32

app = Flask(__name__)

class VideoPublisher(Node):
    def __init__(self, frame_rate=15):
        super().__init__('video_publisher')
        # Correctly initialize the PID publisher
        #self.pid_publisher = self.create_publisher(Int32, '/node_pids', 10)
        #self.publish_pid()
        #self.timer_publish_pid = self.create_timer(2.0, self.publish_pid)  # Setup a timer to call it at regular intervals

        self.publisher = self.create_publisher(Image, 'video', 10)
        self.bridge = CvBridge()
        self.frame_rate = frame_rate
        self.device_path = '/dev/video0'
        self.cap = cv2.VideoCapture(self.device_path)
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera at {self.device_path}.')

    def gen_frames(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            # Adjust frame rate by waiting
            cv2.waitKey(int(1000 / self.frame_rate))
            frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            ret, buffer = cv2.imencode('.jpg', frame_rotated)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def publish_pid(self):
        # Publishes this node's PID to the '/node_pids' topic
        pid_msg = Int32()
        pid_msg.data = os.getpid()
        self.pid_publisher.publish(pid_msg)
        self.get_logger().info(f'Published video_feed PID: {pid_msg.data}')


@app.route('/video_feed')
def video_feed():
    return Response(video_publisher.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask_app():
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

def main(args=None):
    rclpy.init(args=args)
    global video_publisher
    video_publisher = VideoPublisher(frame_rate=15)  # Adjust frame rate as needed
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()
    rclpy.spin(video_publisher)
    video_publisher.cap.release()
    video_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
