import os
import threading
import time
from queue import Queue

import serial
import serial.tools.list_ports


class SerialReaderThread(threading.Thread):
    """Background thread for reading serial data from ESP32"""
    def __init__(self, port=None, baudrate=115200):
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.running = True
        self.message_queue = Queue(maxsize=10)
        
        # Auto-detect ESP32 port if not specified
        if not self.port:
            self.port = self.find_esp32_port()
        
        if self.port:
            try:
                self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.1)
                print(f">>> Serial connected to {self.port} at {self.baudrate} baud")
            except Exception as e:
                print(f">>> Failed to open serial port {self.port}: {e}")
                self.serial_conn = None
        else:
            print(">>> No ESP32 device found")
    
    def find_esp32_port(self):
        """Auto-detect ESP32 COM port (Windows/Linux)"""
        ports = serial.tools.list_ports.comports()
        
        # Check for Linux Bluetooth RFCOMM first
        import platform
        if platform.system() == 'Linux':
            # Check for /dev/rfcomm0 (Bluetooth)
            if os.path.exists('/dev/rfcomm0'):
                print(">>> Found Bluetooth RFCOMM at /dev/rfcomm0")
                return '/dev/rfcomm0'
        
        # Check standard serial ports
        for port in ports:
            # ESP32 identifiers (USB or Bluetooth)
            identifiers = ['USB', 'CH340', 'CP210', 'UART', 'Serial', 'Bluetooth']
            if any(id in port.description for id in identifiers):
                print(f">>> Found potential ESP32 at {port.device}: {port.description}")
                return port.device
        
        return None
    
    def run(self):
        if not self.serial_conn:
            print(">>> Serial reader thread: No connection available")
            return
            
        print(">>> Serial reader thread started")
        while self.running:
            try:
                if self.serial_conn and self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    if line:
                        # Put message in queue
                        if not self.message_queue.full():
                            self.message_queue.put(line)
                        if line == "disable_image":
                            print(f">>> Serial received: {line}")
                time.sleep(0.01)
            except Exception as e:
                print(f">>> Serial read error: {e}")
                time.sleep(0.1)
    
    def get_message(self):
        """Get message from queue (non-blocking)"""
        try:
            return self.message_queue.get_nowait()
        except:
            return None
    
    def stop(self):
        self.running = False
        if self.serial_conn:
            self.serial_conn.close()
        print(">>> Serial reader thread stopped")

class YOLODetectionThread(threading.Thread):
    """Background thread for YOLO inference to prevent GUI blocking"""
    def __init__(self, model, use_bantal_model):
        super().__init__(daemon=True)
        self.model = model
        self.use_bantal_model = use_bantal_model
        self.frame_queue = Queue(maxsize=1)  # Reduce to 1 for less memory
        self.result_queue = Queue(maxsize=1)
        self.running = True
        self.inference_count = 0
        self.last_inference_time = 0
        print(">>> YOLO thread initialized")
        
    def run(self):
        print(">>> YOLO thread started")
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get(timeout=0.01)
                    
                    # Measure inference time
                    start_time = time.time()
                    
                    # YOLO inference
                    if self.use_bantal_model:
                        results = self.model(frame, verbose=False)
                        
                        # Convert Ultralytics results to list format
                        if len(results) > 0 and hasattr(results[0], 'boxes'):
                            boxes = results[0].boxes
                            detections = []
                            
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = box.conf[0].cpu().numpy()
                                cls = int(box.cls[0].cpu().numpy())
                                detections.append([x1, y1, x2, y2, conf, cls, ''])
                            
                            list_tracked_objects = detections
                        else:
                            list_tracked_objects = []
                    else:
                        results = self.model(frame)
                        df_tracked_objects = results.pandas().xyxy[0]
                        list_tracked_objects = df_tracked_objects.values.tolist()
                    
                    self.last_inference_time = (time.time() - start_time) * 1000
                    self.inference_count += 1
                    
                    # Debug output every 30 inferences
                    if self.inference_count % 30 == 0:
                        print(f">>> YOLO: {self.inference_count} inferences, last: {self.last_inference_time:.1f}ms")
                    
                    # Put results in queue (remove old if full)
                    if self.result_queue.full():
                        try:
                            self.result_queue.get_nowait()
                        except:
                            pass
                    self.result_queue.put(list_tracked_objects)
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"YOLO thread error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.01)
    
    def stop(self):
        self.running = False
        print(">>> YOLO thread stopped")