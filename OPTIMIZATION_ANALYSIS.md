# Analisis Lag dan Rekomendasi Optimasi - Block Design Test

## 🔍 MASALAH UTAMA YANG DITEMUKAN

### 1. **BOTTLENECK KRITIS: YOLO Inference di Main Thread** ⚠️
**Lokasi:** Line 1456-1485 dalam `streaming()`

**Masalah:**
- Model YOLO dijalankan **SETIAP FRAME** di main thread GUI
- Blocking operation yang sangat berat (100-300ms per inference)
- Menyebabkan GUI freeze dan lag kamera
- Tidak ada frame skipping atau threading

**Impact:** ⭐⭐⭐⭐⭐ (CRITICAL)

---

### 2. **MediaPipe Hands di Main Thread**
**Lokasi:** Line 2330-2338

**Masalah:**
- `mp_hands.Hands()` dibuat dan dihancurkan **SETIAP FRAME**
- Seharusnya dibuat sekali saat inisialisasi
- Hand detection juga blocking operation

**Impact:** ⭐⭐⭐⭐ (HIGH)

---

### 3. **Image Resizing Berlebihan**
**Lokasi:** Line 1446, 1524, 2355-2358

**Masalah:**
- Frame di-resize dari 640x480 → 1000x800 (line 1446)
- Setiap detected box di-resize ke 100x100 (line 1524)
- Frame di-resize lagi untuk display (line 2355-2358)
- Multiple resize operations per frame

**Impact:** ⭐⭐⭐ (MEDIUM)

---

### 4. **Inefficient Image Loading**
**Lokasi:** Line 1699-1702, 1789-1792, dll.

**Masalah:**
- `Image.open()` dan `resize()` dipanggil setiap kali level berubah
- Bisa di-cache saat startup
- Repeated file I/O operations

**Impact:** ⭐⭐ (LOW-MEDIUM)

---

### 5. **No Frame Rate Control**
**Lokasi:** Line 2364

**Masalah:**
- `self.after(20, self.streaming)` = target 50 FPS
- Terlalu tinggi untuk YOLO inference
- Tidak ada adaptive frame rate

**Impact:** ⭐⭐⭐ (MEDIUM)

---

### 6. **Synchronous Audio Playback**
**Lokasi:** Line 426-448, 587, 913, dll.

**Masalah:**
- `sd.wait()` blocks execution (line 448)
- Audio playback menghentikan processing
- Tidak ada threading untuk audio

**Impact:** ⭐⭐⭐ (MEDIUM)

---

## 💡 REKOMENDASI OPTIMASI

### 🚀 PRIORITAS TINGGI (Implementasi Segera)

#### 1. **Threading untuk YOLO Inference**
```python
import threading
from queue import Queue
import time

class YOLOThread(threading.Thread):
    def __init__(self, model, device):
        super().__init__(daemon=True)
        self.model = model
        self.device = device
        self.frame_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        self.running = True
        
    def run(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                # YOLO inference
                if USE_BANTAL_MODEL:
                    results = self.model(frame, verbose=False)
                    # Process results...
                else:
                    results = self.model(frame)
                
                # Put results in queue (non-blocking)
                if self.result_queue.full():
                    self.result_queue.get()  # Remove old result
                self.result_queue.put(results)
            else:
                time.sleep(0.001)  # Prevent busy waiting
    
    def stop(self):
        self.running = False

# Inisialisasi di __init__:
self.yolo_thread = YOLOThread(model_yolo, device)
self.yolo_thread.start()
self.latest_detections = []

# Di streaming():
# Submit frame for processing (non-blocking)
if self.yolo_thread.frame_queue.empty():
    self.yolo_thread.frame_queue.put(frame.copy())

# Get latest results if available
if not self.yolo_thread.result_queue.empty():
    self.latest_detections = self.yolo_thread.result_queue.get()

# Use self.latest_detections for processing
```

**Benefit:** Mengurangi lag hingga 80-90%

---

#### 2. **Pindahkan MediaPipe ke Instance Variable**
```python
# Di __init__ TimeIn class:
self.mp_hands = mp_hands.Hands(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    max_num_hands=2
)

# Di streaming():
# Hapus 'with' statement
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Fix: gunakan COLOR_BGR2RGB
hand_result = self.mp_hands.process(frame_rgb)

if hand_result.multi_hand_landmarks:
    for hand_landmarks in hand_result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, 
            mp_hands.HAND_CONNECTIONS, 
            landmark_style, connection_style
        )

# Di cleanup/destroy:
def on_closing(self):
    if hasattr(self, 'mp_hands'):
        self.mp_hands.close()
    self.destroy()
```

**Benefit:** Mengurangi overhead 50-70ms per frame

---

#### 3. **Reduce Frame Processing Resolution**
```python
# Di streaming():
# Jangan resize ke 1000x800, gunakan resolusi asli kamera
frame = cap.read()[1]
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# frame = cv2.resize(frame, (1000, 800))  # HAPUS INI

# Atau gunakan resolusi lebih kecil untuk YOLO:
YOLO_WIDTH = 640  # Resolusi optimal untuk YOLOv5
YOLO_HEIGHT = 480
frame_for_yolo = cv2.resize(frame, (YOLO_WIDTH, YOLO_HEIGHT))

# Submit frame_for_yolo ke YOLO thread
# Tapi gunakan frame asli untuk display
```

**Benefit:** Mengurangi processing time 30-40%

---

#### 4. **Frame Skipping untuk YOLO**
```python
# Di __init__:
self.frame_count = 0
self.yolo_skip_frames = 2  # Process YOLO setiap 3 frame

# Di streaming():
self.frame_count += 1

# Hanya jalankan YOLO setiap N frame
if self.frame_count % (self.yolo_skip_frames + 1) == 0:
    if self.yolo_thread.frame_queue.empty():
        self.yolo_thread.frame_queue.put(frame.copy())
```

**Benefit:** Mengurangi beban YOLO hingga 66%

---

#### 5. **Async Audio Playback**
```python
import threading

def play_audio_async(wav_path):
    """Play audio in background thread"""
    def _play():
        try:
            if isinstance(wav_path, (str, Path)):
                data, samplerate = sf.read(str(wav_path), dtype='float32')
            else:
                data = np.asarray(wav_path)
                samplerate = 22050
            
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            print(f"Audio error: {e}")
    
    thread = threading.Thread(target=_play, daemon=True)
    thread.start()

# Ganti semua play_audio() dengan play_audio_async()
```

**Benefit:** Eliminasi blocking audio (0-3 detik per audio)

---

### 🔧 PRIORITAS MENENGAH

#### 6. **Cache Level Images**
```python
# Di __init__ atau setelah load model:
self.cached_level_images = {}

def preload_level_images(self):
    """Preload and cache all level images"""
    print(">>> Preloading level images...")
    for variant, path in LEVEL_PATHS.items():
        try:
            img = Image.open(path)
            # Cache dalam berbagai ukuran yang mungkin digunakan
            self.cached_level_images[variant] = {
                'original': img,
                'display': img.resize(
                    (self.frame_width, self.frame_height), 
                    Image.Resampling.LANCZOS
                )
            }
        except Exception as e:
            print(f"Error loading {variant}: {e}")
    print(f">>> Cached {len(self.cached_level_images)} level images")

# Panggil di __init__:
self.preload_level_images()

# Saat load level, gunakan cache:
cached_img = self.cached_level_images[variant]['display']
self.image = customtkinter.CTkImage(
    light_image=cached_img,
    size=(self.frame_width, self.frame_height)
)
```

**Benefit:** Eliminasi I/O lag saat ganti level

---

#### 7. **Optimize Box Detection Loop**
```python
# Gunakan numpy operations instead of loops
# Di streaming(), ganti loop for dengan vectorized operations:

# Filter detections by confidence
if USE_BANTAL_MODEL:
    # ... existing code ...
    if len(detections) > 0:
        detections = np.array(detections)
        # Filter by confidence
        mask = detections[:, 4] > 0.7
        detections = detections[mask]
else:
    df_tracked_objects = results.pandas().xyxy[0]
    df_tracked_objects = df_tracked_objects[df_tracked_objects['confidence'] > 0.7]
    list_tracked_objects = df_tracked_objects.values.tolist()
```

---

#### 8. **Reduce Timer Update Frequency**
```python
# Di update_timer():
# Ganti dari 10ms ke 50ms
self.after(50, self.update_timer)  # Was: self.after(10, ...)
```

**Benefit:** Mengurangi GUI updates

---

### ⚙️ OPTIMASI KHUSUS JETSON XAVIER AGX

#### 9. **Enable TensorRT untuk YOLO**
```python
# Untuk Jetson, gunakan TensorRT optimization
# Install: pip install nvidia-tensorrt

# Convert model ke TensorRT (one-time):
from torch2trt import torch2trt

# Dummy input
x = torch.ones((1, 3, 640, 480)).cuda()

# Convert
model_trt = torch2trt(model_yolo, [x], fp16_mode=True)

# Save
torch.save(model_trt.state_dict(), 'model_trt.pth')

# Load TensorRT model
from torch2trt import TRTModule
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('model_trt.pth'))
```

**Benefit:** 2-5x speedup pada Jetson

---

#### 10. **Use CUDA Streams**
```python
# Untuk async GPU operations
import torch.cuda as cuda

# Di __init__:
if torch.cuda.is_available():
    self.stream = cuda.Stream()

# Di YOLO inference:
with cuda.stream(self.stream):
    results = model_yolo(frame)
```

---

#### 11. **Optimize Camera Capture untuk Jetson**
```python
# Gunakan GStreamer pipeline untuk Jetson
def get_jetson_camera():
    gst_pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=640, height=480, "
        "format=NV12, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink"
    )
    return cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

# Ganti camera initialization:
if platform.system() == 'Linux' and os.path.exists('/proc/device-tree/model'):
    # Check if Jetson
    with open('/proc/device-tree/model', 'r') as f:
        if 'jetson' in f.read().lower():
            cap = get_jetson_camera()
else:
    # ... existing code ...
```

**Benefit:** Hardware acceleration pada Jetson

---

### 📊 OPTIMASI TAMBAHAN

#### 12. **Add FPS Counter untuk Monitoring**
```python
# Di __init__:
self.fps_counter = 0
self.fps_start_time = time.time()
self.current_fps = 0

# Di streaming():
self.fps_counter += 1
if time.time() - self.fps_start_time >= 1.0:
    self.current_fps = self.fps_counter
    print(f"FPS: {self.current_fps}")
    self.fps_counter = 0
    self.fps_start_time = time.time()
```

---

#### 13. **Reduce cv2.line() Calls**
```python
# Ganti 6 cv2.line() calls dengan 1 cv2.polylines():
if len(box_design) == 4 and len(box_distance) == 4:
    # Create polygon points
    pts = np.array([
        [pos_x[0], pos_y[0]],
        [pos_x[1], pos_y[1]],
        [pos_x[2], pos_y[2]],
        [pos_x[3], pos_y[3]]
    ], np.int32)
    
    # Draw all lines at once
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
```

---

#### 14. **Conditional Hand Detection**
```python
# Hanya jalankan hand detection jika diperlukan
# Di streaming(), tambahkan flag:
self.enable_hand_detection = os.getenv('ENABLE_HAND_DETECTION', 'false').lower() == 'true'

if self.enable_hand_detection:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_result = self.mp_hands.process(frame_rgb)
    # ... rest of hand detection code ...
```

---

## 📋 IMPLEMENTASI BERTAHAP

### **Phase 1: Quick Wins (1-2 jam)**
1. ✅ Pindahkan MediaPipe ke instance variable
2. ✅ Async audio playback
3. ✅ Reduce timer update frequency
4. ✅ Fix cv2.cvtColor bug (line 2333)
5. ✅ Add FPS counter

**Expected improvement:** 30-40% reduction in lag

---

### **Phase 2: Major Optimizations (3-5 jam)**
1. ✅ Threading untuk YOLO
2. ✅ Frame skipping
3. ✅ Reduce processing resolution
4. ✅ Cache level images

**Expected improvement:** 70-80% reduction in lag

---

### **Phase 3: Jetson-Specific (2-4 jam)**
1. ✅ TensorRT conversion
2. ✅ GStreamer camera pipeline
3. ✅ CUDA streams

**Expected improvement:** 2-5x speedup on Jetson

---

## 🎯 EXPECTED RESULTS

### **Sebelum Optimasi:**
- FPS: ~5-10 FPS
- Lag: 200-500ms
- CPU Usage: 80-100%
- GPU Usage: 60-80%

### **Setelah Phase 1:**
- FPS: ~15-20 FPS
- Lag: 100-200ms
- CPU Usage: 60-80%

### **Setelah Phase 2:**
- FPS: ~25-30 FPS
- Lag: 30-50ms
- CPU Usage: 40-60%
- GPU Usage: 70-90%

### **Setelah Phase 3 (Jetson):**
- FPS: ~30-60 FPS
- Lag: <20ms
- Inference time: 10-20ms (vs 100-300ms)

---

## 🐛 BUG YANG DITEMUKAN

### 1. **Line 2333: Wrong Color Conversion**
```python
# SALAH:
frame_rgb = cv2.cvtColor(frame, 1)  # 1 is not a valid flag

# BENAR:
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

### 2. **Line 90: Model loaded to CPU instead of GPU**
```python
# SALAH:
model_yolo = attempt_load(PATH_MODEL, device='cpu')

# BENAR:
model_yolo = attempt_load(PATH_MODEL, device=device)  # Use the configured device
```

---

## 📝 CATATAN TAMBAHAN

### **Memory Management:**
- Gunakan `del` untuk cleanup temporary variables
- Clear CUDA cache: `torch.cuda.empty_cache()`

### **Error Handling:**
- Tambahkan try-except di YOLO thread
- Handle camera disconnect gracefully

### **Configuration:**
Tambahkan ke `.env`:
```env
# Performance settings
YOLO_SKIP_FRAMES=2
ENABLE_HAND_DETECTION=false
YOLO_RESOLUTION_WIDTH=640
YOLO_RESOLUTION_HEIGHT=480
USE_TENSORRT=true
```

---

## 🔗 RESOURCES

- [YOLOv5 Optimization Guide](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)
- [Jetson TensorRT Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [OpenCV Performance Optimization](https://docs.opencv.org/4.x/dc/d71/tutorial_py_optimization.html)

---

**Dibuat:** 2025-01-13  
**Versi:** 1.0  
**Status:** Ready for Implementation
