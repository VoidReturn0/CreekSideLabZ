import threading
import multiprocessing
from queue import Queue
import time
import logging

logging.basicConfig(level=logging.DEBUG)

# Shared queue for inter-thread and inter-process communication
shared_queue = Queue()

# Function to simulate work done by a thread
def threaded_function(arg, shared_q):
    logging.info(f'Thread started with argument: {arg}')
    time.sleep(2)
    shared_q.put(f'Thread done with argument: {arg}')

# Function to simulate work done by a process
def process_function(arg, shared_q):
    logging.info(f'Process started with argument: {arg}')
    time.sleep(2)
    shared_q.put(f'Process done with argument: {arg}')

if __name__ == '__main__':
    # Start the thread
    thread = threading.Thread(target=threaded_function, args=(1, shared_queue))
    thread.start()
    
    # Start the process
    process = multiprocessing.Process(target=process_function, args=(2, shared_queue))
    process.start()
    
    # Wait for both to finish
    thread.join()
    process.join()
    
    # Read their statuses from the shared queue
    while not shared_queue.empty():
        print(shared_queue.get())
