#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Import

import numpy as np
import cv2                                                                        
import os
import math
import time
import sys
import threading
import traceback
from pathlib import Path
from config import (
    BASE_DIR, DISPLAY_HALF, BUTTON_MODE, HIDE_CAMERA, SHOW_ROTATION_COUNT,
    MAX_LEVEL, LEVEL_COUNT, COUNTDOWN_SECONDS, CUSTOM_LEVELS, LEVEL_PATHS, LEVEL_ANSWERS,
)  # config memuat .env: harus diimpor sebelum modul lain yang membaca env (mis. audio)
from scoring import estimate_cognitive_age, average_cognitive_age
from scoring import cognitive_fitness as fitness_from_ages
from eyetracker_module import EyeTrackerHandler
from aoi import AoiTracker, build_rects
from rotation_tracker import BlockRotationTracker, FACE_STATE_MAP, DIAG_CYCLE
from workers import YOLODetectionThread, SerialReaderThread

from logging_setup import install_exception_hooks, log_exception
install_exception_hooks()  

from audio import play_audio 

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Learning

import torch
 
# Device configuration
print("GPU CUDA is Available : " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

USE_BANTAL_MODEL = os.getenv('MODEL_BANTAL', 'false').lower() == 'true'
if USE_BANTAL_MODEL:
    PATH_MODEL = os.path.join(BASE_DIR, 'MODEL', 'bantal', 'bantal.pt')
    
    try:
        print(">>> Loading bantal.pt with Ultralytics format...")
        
        # Cek apakah ini model Ultralytics (v8+ atau v7.0+)
        checkpoint = torch.load(PATH_MODEL, map_location='cpu', weights_only=False)
        
        if 'version' in checkpoint:
            print(f">>> Model version: {checkpoint['version']}")
        
        # Jika model dari Ultralytics (punya key 'version')
        if 'version' in checkpoint and hasattr(checkpoint['model'], '__class__') and \
           'ultralytics' in str(checkpoint['model'].__class__):
            
            print(">>> Detected Ultralytics format, loading with ultralytics library...")
            
            # Coba load dengan ultralytics library
            try:
                from ultralytics import YOLO
                model_yolo = YOLO(PATH_MODEL)
                model_yolo.to(device)
                print(">>> Model bantal.pt loaded successfully with Ultralytics YOLO")
            except ImportError:
                print(">>> Ultralytics library not found, installing...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
                from ultralytics import YOLO
                model_yolo = YOLO(PATH_MODEL)
                model_yolo.to(device)
                print(">>> Model bantal.pt loaded successfully with Ultralytics YOLO")
        
        else:
            # Fallback: gunakan attempt_load biasa
            print(">>> Loading with attempt_load...")
            import sys
            sys.path.insert(0, os.path.join(BASE_DIR, 'MODEL', 'yolov5'))
            from models.experimental import attempt_load
            
            model_yolo = attempt_load(PATH_MODEL, device=device)
            
            # Inisialisasi grid untuk model lama
            for m in model_yolo.modules():
                if type(m).__name__ == 'Detect':
                    if not hasattr(m, 'grid'):
                        m.grid = [torch.zeros(1)] * m.nl
                    if not hasattr(m, 'anchor_grid'):
                        m.anchor_grid = [torch.zeros(1)] * m.nl
            
            model_yolo.to(device)
            print(">>> Model bantal.pt loaded successfully with attempt_load")
    
    except Exception as e:
        print(f">>> Failed to load bantal.pt: {e}")
        print(">>> Falling back to best.pt...")
        USE_BANTAL_MODEL = False
        PATH_MODEL = os.path.join(BASE_DIR, 'MODEL', 'exp7', 'weights', 'best.pt')

if not USE_BANTAL_MODEL:
    PATH_MODEL = os.path.join(BASE_DIR, 'MODEL', 'exp7', 'weights', 'best.pt')
    model_yolo = torch.hub.load(
        os.path.join(BASE_DIR, 'MODEL', 'yolov5'), 
        'custom', 
        path=PATH_MODEL, 
        force_reload=True, 
        source='local'
    )
    model_yolo.to(device)

# Random Level
import random
face_01 = cv2.imread(os.path.join(BASE_DIR, 'FILES', 'LABEL_50x50', '01.png'), cv2.IMREAD_COLOR)
face_02 = cv2.imread(os.path.join(BASE_DIR, 'FILES', 'LABEL_50x50', '02.png'), cv2.IMREAD_COLOR)
face_03 = cv2.imread(os.path.join(BASE_DIR, 'FILES', 'LABEL_50x50', '03.png'), cv2.IMREAD_COLOR)
face_04 = cv2.imread(os.path.join(BASE_DIR, 'FILES', 'LABEL_50x50', '04.png'), cv2.IMREAD_COLOR)
face_05 = cv2.imread(os.path.join(BASE_DIR, 'FILES', 'LABEL_50x50', '05.png'), cv2.IMREAD_COLOR)
face_06 = cv2.imread(os.path.join(BASE_DIR, 'FILES', 'LABEL_50x50', '06.png'), cv2.IMREAD_COLOR)

gray_face_01 = cv2.cvtColor(face_01, cv2.COLOR_BGR2GRAY)
gray_face_02 = cv2.cvtColor(face_02, cv2.COLOR_BGR2GRAY)
gray_face_03 = cv2.cvtColor(face_03, cv2.COLOR_BGR2GRAY)
gray_face_04 = cv2.cvtColor(face_04, cv2.COLOR_BGR2GRAY)
gray_face_05 = cv2.cvtColor(face_05, cv2.COLOR_BGR2GRAY)
gray_face_06 = cv2.cvtColor(face_06, cv2.COLOR_BGR2GRAY)

ret_01, mask_face_01 = cv2.threshold(gray_face_01, 1, 255, cv2.THRESH_BINARY)
ret_02, mask_face_02 = cv2.threshold(gray_face_02, 1, 255, cv2.THRESH_BINARY)
ret_03, mask_face_03 = cv2.threshold(gray_face_03, 1, 255, cv2.THRESH_BINARY)
ret_04, mask_face_04 = cv2.threshold(gray_face_04, 1, 255, cv2.THRESH_BINARY)
ret_05, mask_face_05 = cv2.threshold(gray_face_05, 1, 255, cv2.THRESH_BINARY)
ret_06, mask_face_06 = cv2.threshold(gray_face_06, 1, 255, cv2.THRESH_BINARY)

img_00 = cv2.imread(os.path.join(BASE_DIR, 'FILES', 'TEST_100x100', '00.jpg'))
img_01 = cv2.imread(os.path.join(BASE_DIR, 'FILES', 'TEST_100x100', '01.jpg'))
img_02 = cv2.imread(os.path.join(BASE_DIR, 'FILES', 'TEST_100x100', '02.jpg'))
img_03 = cv2.imread(os.path.join(BASE_DIR, 'FILES', 'TEST_100x100', '03.jpg'))
img_04 = cv2.imread(os.path.join(BASE_DIR, 'FILES', 'TEST_100x100', '04.jpg'))
img_05 = cv2.imread(os.path.join(BASE_DIR, 'FILES', 'TEST_100x100', '05.jpg'))
img_06 = cv2.imread(os.path.join(BASE_DIR, 'FILES', 'TEST_100x100', '06.jpg'))
img_07 = cv2.imread(os.path.join(BASE_DIR, 'FILES', 'TEST_100x100', '07.jpg'))
img_08 = cv2.imread(os.path.join(BASE_DIR, 'FILES', 'TEST_100x100', '08.jpg'))
img_09 = cv2.imread(os.path.join(BASE_DIR, 'FILES', 'TEST_100x100', '09.jpg'))

gray_img_00 = cv2.cvtColor(img_00, cv2.COLOR_BGR2GRAY)
gray_img_01 = cv2.cvtColor(img_01, cv2.COLOR_BGR2GRAY)
gray_img_02 = cv2.cvtColor(img_02, cv2.COLOR_BGR2GRAY)
gray_img_03 = cv2.cvtColor(img_03, cv2.COLOR_BGR2GRAY)
gray_img_04 = cv2.cvtColor(img_04, cv2.COLOR_BGR2GRAY)
gray_img_05 = cv2.cvtColor(img_05, cv2.COLOR_BGR2GRAY)
gray_img_06 = cv2.cvtColor(img_06, cv2.COLOR_BGR2GRAY)
gray_img_07 = cv2.cvtColor(img_07, cv2.COLOR_BGR2GRAY)
gray_img_08 = cv2.cvtColor(img_08, cv2.COLOR_BGR2GRAY)
gray_img_09 = cv2.cvtColor(img_09, cv2.COLOR_BGR2GRAY)

ret_img_00, mask_img_00 = cv2.threshold(gray_img_00, 1, 255, cv2.THRESH_BINARY)
ret_img_01, mask_img_01 = cv2.threshold(gray_img_01, 1, 255, cv2.THRESH_BINARY)
ret_img_02, mask_img_02 = cv2.threshold(gray_img_02, 1, 255, cv2.THRESH_BINARY)
ret_img_03, mask_img_03 = cv2.threshold(gray_img_03, 1, 255, cv2.THRESH_BINARY)
ret_img_04, mask_img_04 = cv2.threshold(gray_img_04, 1, 255, cv2.THRESH_BINARY)
ret_img_05, mask_img_05 = cv2.threshold(gray_img_05, 1, 255, cv2.THRESH_BINARY)
ret_img_06, mask_img_06 = cv2.threshold(gray_img_06, 1, 255, cv2.THRESH_BINARY)
ret_img_07, mask_img_07 = cv2.threshold(gray_img_07, 1, 255, cv2.THRESH_BINARY)
ret_img_08, mask_img_08 = cv2.threshold(gray_img_08, 1, 255, cv2.THRESH_BINARY)
ret_img_09, mask_img_09 = cv2.threshold(gray_img_09, 1, 255, cv2.THRESH_BINARY)

FACE_OVERLAYS = {
    1: (mask_face_01, face_01), 2: (mask_face_02, face_02),
    3: (mask_face_03, face_03), 4: (mask_face_04, face_04),
    5: (mask_face_05, face_05), 6: (mask_face_06, face_06),
}


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Connect to Database Server 
  
# PostgreSQL Setup
import psycopg2

def get_db_connection():
    """Create and return a PostgreSQL database connection using environment variables."""
    try:
        # Get database configuration from environment variables
        db_config = {
            'dbname': os.getenv('DB_DATABASE'),
            'user': os.getenv('DB_USERNAME'),
            'password': os.getenv('DB_PASSWORD'),
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT')
        }
        
        # Establish connection
        conn = psycopg2.connect(**db_config)
        conn.autocommit = True
        
        # Set timezone to UTC+7 (Western Indonesia Time)
        with conn.cursor() as cursor:
            cursor.execute("SET timezone = 'Asia/Jakarta';")
            
        print(">>> PostgreSQL database connected successfully.")
        print(">>> Timezone set to Asia/Jakarta (UTC+7)")
        return conn
        
    except Exception as e:
        print(">>> Error connecting to PostgreSQL database:")
        print(f"Error: {e}")
        return None

# Example usage:
# conn = get_db_connection()
# if conn:
#     with conn.cursor() as cursor:
#         cursor.execute("SELECT version();")
#         db_version = cursor.fetchone()
#         print(f"PostgreSQL database version: {db_version[0]}")
#     conn.close()


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SQL File Executor Background Thread

# Background thread untuk auto-execute SQL files yang di-generate oleh aplikasi
def sql_executor_worker():
    """Background worker yang berjalan setiap 5 detik untuk execute SQL files"""
    try:
        # Import SQL executor module
        import sys
        sys.path.insert(0, os.path.join(BASE_DIR, 'cron'))
        from execute_sql import process_sql_files
        
        while True:
            try:
                # Process SQL files dengan silent mode (no log jika tidak ada file)
                sql_dir = Path(BASE_DIR) / 'cron' / 'sql'
                if sql_dir.exists() and list(sql_dir.glob('*.sql')):
                    # Ada file SQL, process
                    process_sql_files(get_db_connection_func=get_db_connection, sql_dir=sql_dir)
                # Sleep 5 detik sebelum check lagi
                time.sleep(5)
            except Exception as e:
                print(f">>> SQL Executor error: {e}")
                time.sleep(5)  # Tetap sleep meskipun error
    except Exception as e:
        print(f">>> SQL Executor thread failed to start: {e}")
        import traceback
        traceback.print_exc()

# Start SQL executor thread saat aplikasi dimulai
sql_executor_thread = threading.Thread(target=sql_executor_worker, daemon=True, name="SQLExecutor")
sql_executor_thread.start()
print(">>> SQL Executor background thread started (checking every 5 seconds)")

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Text to  

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    print(f">>> MediaPipe import failed: {e}. Hand landmark detection disabled.")

mp_drawing = None
mp_hands = None
landmark_style = None
connection_style = None

if mp is not None:
    try:
        mp_solutions = getattr(mp, "solutions", None)
        if mp_solutions is None:
            from mediapipe.python import solutions as mp_solutions
        mp_drawing = mp_solutions.drawing_utils
        mp_hands = mp_solutions.hands
        landmark_style = mp_drawing.DrawingSpec(color=(0,0,255), thickness=3, circle_radius=3)
        connection_style = mp_drawing.DrawingSpec(color=(0,0,255), thickness=4)
    except Exception as e:
        print(f">>> MediaPipe 'solutions' API unavailable: {e}. Hand landmark detection disabled.")



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Setup GUI Input 

import customtkinter
from PIL import Image, ImageTk
import cv2

customtkinter.set_appearance_mode("System")         # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

# Input Data
nick_name       = "NULL"
gender_code     = "NULL"
age_range_code  = "NULL"

class MyTextboxFrame(customtkinter.CTkFrame):
    def __init__(self, master, title, values):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.values = values
        self.title = title

        self.title = customtkinter.CTkLabel(self, text=self.title, fg_color="gray", text_color="white", font=("Helvetica",30), corner_radius=6)
        self.title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")

        self.textbox = customtkinter.CTkTextbox(master=self, width=100, height=10, corner_radius=0, font=("Helvetica",50))
        self.textbox.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="nsew")

    def get(self):
        str_name = self.textbox.get('0.0', '10.0')
        str_name = str_name.rstrip()
        return str_name


class MyRadiobuttonFrame(customtkinter.CTkFrame):
    def __init__(self, master, title, values):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.values = values
        self.title = title
        self.radiobuttons = []
        self.variable = customtkinter.StringVar(value="")

        self.title = customtkinter.CTkLabel(self, text=self.title, fg_color="gray", text_color="white", font=("Helvetica",30), corner_radius=6)
        self.title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")

        for i, value in enumerate(self.values):
            radiobutton = customtkinter.CTkRadioButton(self, text=value, value=value, variable=self.variable, font=("Helvetica",30))
            radiobutton.grid(row=i + 1, column=0, padx=10, pady=(10, 0), sticky="w")
            self.radiobuttons.append(radiobutton)

    def get(self):
        return self.variable.get()

    def set(self, value):
        self.variable.set(value)


class App_Input(customtkinter.CTk): #CTkToplevel
    def __init__(self):
        super().__init__()

        self.title("User Input")
        
        spawn_x = 600           # int((self.winfo_screenwidth()-480)/2) 
        spawn_y = 10            # int((self.winfo_screenheight()-320)/2)
        self.geometry(f"{480}x{320}+{spawn_x}+{spawn_y}")
        #self.geometry("480x320")
        
        # from AppOpener import open
        # open("on-screen keyboard")

        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.resizable(0,0)

        self.textbox_frame_0 = MyTextboxFrame(self, "Nick Name", values=" ")
        self.textbox_frame_0.grid(row=0, column=0, padx=10, columnspan=2, pady=(5, 0), sticky="nsew")

        self.textbox_frame_1 = MyTextboxFrame(self, "Age", values=" ")
        self.textbox_frame_1.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="nsew")      
        
        self.radiobutton_frame_0 = MyRadiobuttonFrame(self, "Gender", values=["male", "female"])
        self.radiobutton_frame_0.grid(row=1, column=1, padx=(0, 10), pady=(10, 0), sticky="nsew")

        self.button = customtkinter.CTkButton(self, text="SUBMIT", command=self.button_callback, font=("Helvetica",30))
        self.button.grid(row=2, column=0, padx=10, pady=10, sticky="ew", columnspan=2)

    def report_callback_exception(self, exc, val, tb):
        """Capture tkinter callback exceptions with full traceback."""
        log_exception("Tk callback exception (App_Input)", exc, val, tb)

    def button_callback(self):
        import tkinter.messagebox as messagebox
        try:
            # Get values from UI
            name = self.textbox_frame_0.get().strip()
            age = self.textbox_frame_1.get().strip()
            gender = self.radiobutton_frame_0.get()

            # Input validation
            if not name:
                raise ValueError("Nama tidak boleh kosong")
            if not age.isdigit():
                raise ValueError("Usia harus berupa angka")
            if not gender:
                raise ValueError("Silakan pilih jenis kelamin")

            # Convert age to integer
            age_int = int(age)
            
            # Debug print
            print("Name   :", name)
            print("Age    :", age_int)
            print("Gender :", gender)

            # Set global variables
            global nick_name, gender_code, age_range_code
            nick_name = name
            age_range_code = age_int #str(age_int)  # Store as string if needed for display
            gender_code = gender

            # Close the window
            self.destroy()

        except ValueError as e:
            # Show error message to user
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            print(f"Unexpected error: {e}")
            messagebox.showerror("Error", "Terjadi kesalahan saat memproses input")
        

try:
    app_input = App_Input()
    app_input.mainloop()
except Exception:
    print(">>> Failed while running input window. Full traceback:")
    traceback.print_exc()
    raise SystemExit(1)




#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Setup GUI Video

#PATH_VIDEO = r"C:\Users\50575\Desktop\BDT\VIDEO_EXPERIMENT\TLL\TOP_VIEW\anom_02.mp4"
#cap = cv2.VideoCapture(PATH_VIDEO)          #0, cv2.CAP_DSHOW
import platform
if platform.system() == 'Windows':
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:  # Linux/Mac
    cap = cv2.VideoCapture(0)  # atau cv2.CAP_V4L2

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)      #1280   640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)     #720    480

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Camera & Detection Settings

from camera_settings import (
    camera_settings, CAMERA_DEFAULTS, DETECT_DEFAULTS,
    save_camera_settings, apply_camera_settings,
)

print(">>> Start Test ...")

class TextFrame(customtkinter.CTkFrame):
    
    def __init__(self, master, title): #, values
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.title = title
        self.title = customtkinter.CTkLabel(self, text=self.title, fg_color="gray30", corner_radius=6)
        self.title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")

    def get(self):
        return self.variable.get()



class ImageFrame(customtkinter.CTkFrame):

    def __init__(self, master, title):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.title = title
        self.title = customtkinter.CTkLabel(self, text=self.title, corner_radius=6) #fg_color="gray30",
        self.title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")

    def get(self):
        return self.variable.get()



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Camera Settings Dialog

class CameraSettingsDialog(customtkinter.CTkToplevel):
    """Setelan kamera (brightness, zoom, orientasi) dengan preview langsung sebelum tes dimulai."""
    PREVIEW_W, PREVIEW_H = 400, 300

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self._after_id = None
        self._preview_img = None
        self.title("Camera Settings")
        self.geometry("440x740")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.transient(app)
        self.grid_columnconfigure(0, weight=1)

        self.preview = customtkinter.CTkLabel(self, text="", width=self.PREVIEW_W, height=self.PREVIEW_H)
        self.preview.grid(row=0, column=0, padx=20, pady=(20, 5))
        self.status = customtkinter.CTkLabel(self, text="", font=("Helvetica", 14))
        self.status.grid(row=1, column=0, padx=20, pady=(0, 10))

        # Brightness
        self.lbl_brightness = customtkinter.CTkLabel(self, text="", font=("Helvetica", 16))
        self.lbl_brightness.grid(row=2, column=0, padx=20, pady=(5, 0), sticky="w")
        self.sl_brightness = customtkinter.CTkSlider(
            self, from_=-100, to=100, number_of_steps=200, command=self._on_brightness)
        self.sl_brightness.grid(row=3, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Zoom
        self.lbl_zoom = customtkinter.CTkLabel(self, text="", font=("Helvetica", 16))
        self.lbl_zoom.grid(row=4, column=0, padx=20, pady=(5, 0), sticky="w")
        self.sl_zoom = customtkinter.CTkSlider(
            self, from_=1.0, to=3.0, number_of_steps=20, command=self._on_zoom)
        self.sl_zoom.grid(row=5, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Orientation
        customtkinter.CTkLabel(self, text="Orientation", font=("Helvetica", 16)).grid(
            row=6, column=0, padx=20, pady=(5, 0), sticky="w")
        self.seg_rotation = customtkinter.CTkSegmentedButton(
            self, values=["0°", "90°", "180°", "270°"], command=self._on_rotation)
        self.seg_rotation.grid(row=7, column=0, padx=20, pady=(0, 8), sticky="ew")
        self.var_flip_h = customtkinter.BooleanVar(value=False)
        self.var_flip_v = customtkinter.BooleanVar(value=False)
        customtkinter.CTkCheckBox(self, text="Mirror (horizontal)", variable=self.var_flip_h,
                                  command=self._on_flip).grid(row=8, column=0, padx=20, pady=2, sticky="w")
        customtkinter.CTkCheckBox(self, text="Flip (vertical)", variable=self.var_flip_v,
                                  command=self._on_flip).grid(row=9, column=0, padx=20, pady=2, sticky="w")

        customtkinter.CTkLabel(
            self, justify="left", wraplength=390, text_color="gray70",
            text="Tip: green boxes are detected blocks. Adjust until all 4 faces are read "
                 "(red faces must look dark, white faces bright). Orientation must match the table "
                 "as the player sees it."
        ).grid(row=10, column=0, padx=20, pady=(10, 5), sticky="w")

        buttons = customtkinter.CTkFrame(self, fg_color="transparent")
        buttons.grid(row=11, column=0, padx=20, pady=(5, 15), sticky="ew")
        buttons.grid_columnconfigure((0, 1), weight=1)
        customtkinter.CTkButton(buttons, text="Reset", fg_color="gray30", hover_color="gray40",
                                command=self.reset).grid(row=0, column=0, padx=(0, 5), sticky="ew")
        customtkinter.CTkButton(buttons, text="Save & Close",
                                command=self.close).grid(row=0, column=1, padx=(5, 0), sticky="ew")

        self._sync_widgets()
        self._refresh()
        self.after(200, self.lift)

    # ---- sinkronisasi widget <-> camera_settings ----
    def _sync_widgets(self):
        s = camera_settings
        self.sl_brightness.set(s["brightness"])
        self.sl_zoom.set(s["zoom"])
        self.seg_rotation.set(f"{s['rotation']}°")
        self.var_flip_h.set(s["flip_h"])
        self.var_flip_v.set(s["flip_v"])
        self.lbl_brightness.configure(text=f"Brightness: {s['brightness']:+d}")
        self.lbl_zoom.configure(text=f"Zoom: {s['zoom']:.1f}x")

    def _on_brightness(self, value):
        camera_settings["brightness"] = int(round(value))
        self.lbl_brightness.configure(text=f"Brightness: {camera_settings['brightness']:+d}")

    def _on_zoom(self, value):
        camera_settings["zoom"] = round(float(value), 1)
        self.lbl_zoom.configure(text=f"Zoom: {camera_settings['zoom']:.1f}x")

    def _on_rotation(self, value):
        camera_settings["rotation"] = int(str(value).replace("°", ""))

    def _on_flip(self):
        camera_settings["flip_h"] = bool(self.var_flip_h.get())
        camera_settings["flip_v"] = bool(self.var_flip_v.get())

    def reset(self):
        camera_settings.update(CAMERA_DEFAULTS)
        self._sync_widgets()

    # ---- preview langsung dari streaming() ----
    def _refresh(self):
        try:
            if not self.winfo_exists():
                return
            frame = self.app.cam_preview_frame
            if frame is not None:
                h, w = frame.shape[:2]
                scale = min(self.PREVIEW_W / w, self.PREVIEW_H / h)
                size = (max(1, int(w * scale)), max(1, int(h * scale)))
                img = Image.fromarray(frame).resize(size, Image.Resampling.BILINEAR)
                self._preview_img = customtkinter.CTkImage(light_image=img, size=size)
                self.preview.configure(image=self._preview_img)
            found, read = self.app.cam_stats
            self.status.configure(text=f"Blocks detected: {found}   |   Faces read: {read}")
            self._after_id = self.after(100, self._refresh)
        except Exception:
            pass  # dialog sedang ditutup

    def close(self):
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        save_camera_settings()
        self.app.cam_dialog = None
        self.destroy()


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Detection Check Dialog

FACE_NAMES = {0: "unreadable", 1: "Red", 2: "White", 3: "Diag (light BL)", 4: "Diag (light TL)",
              5: "Diag (light TR)", 6: "Diag (light BR)"}
OK_COLOR, BAD_COLOR, DIM_COLOR = "#2ecc71", "#e74c3c", "gray70"


def blocks_form_square(points):
    """Sama dengan syarat di pipeline: 4 balok, empat jarak terpendek (sisi persegi) selisihnya < 100 px."""
    if len(points) != 4:
        return False
    lens = sorted(math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])
                  for i in range(4) for j in range(i + 1, 4))
    return all(abs(lens[i] - lens[j]) < 100 for i in range(4) for j in range(i + 1, 4))


def describe_tracker_event(ev):
    _, _, tid, old, new, kind = ev
    o, n = FACE_NAMES.get(old, "?"), FACE_NAMES.get(new, "?")
    if kind == "rotate":
        step = (DIAG_CYCLE.index(new) - DIAG_CYCLE.index(old)) % 4
        how = {1: "90 deg CW", 2: "180 deg", 3: "90 deg CCW"}[step]
        return f"Block #{tid}: {o} -> {n}  (rotate {how})"
    if kind == "flip":
        return f"Block #{tid}: {o} -> {n}  (flip)"
    return f"Block #{tid}: lifted, same face ({o})  (lift, no visible change)"


class DetectionCheckDialog(customtkinter.CTkToplevel):
    """Cek deteksi balok dan tes deteksi flip/rotasi sebelum tes dimulai, plus penyetel ambang deteksi."""
    PREVIEW_W, PREVIEW_H = 400, 300

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self._after_id = None
        self._preview_img = None
        self.title("Detection Check")
        self.geometry("440x730")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.transient(app)
        self.grid_columnconfigure(0, weight=1)

        self.preview = customtkinter.CTkLabel(self, text="", width=self.PREVIEW_W, height=self.PREVIEW_H)
        self.preview.grid(row=0, column=0, padx=20, pady=(15, 5))
        self.verdict = customtkinter.CTkLabel(self, text="", font=("Helvetica", 18, "bold"))
        self.verdict.grid(row=1, column=0, padx=20, pady=(0, 8))

        tabs = customtkinter.CTkTabview(self, width=400, height=300)
        tabs.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="nsew")
        tab_blocks = tabs.add("Blocks")
        tab_motion = tabs.add("Flip / Rotation")
        tab_tuning = tabs.add("Tuning")

        # ---- Tab Blocks: checklist + daftar deteksi ----
        self.check_labels = []
        for i in range(4):
            lbl = customtkinter.CTkLabel(tab_blocks, text="", anchor="w", font=("Helvetica", 15))
            lbl.grid(row=i, column=0, padx=10, pady=1, sticky="w")
            self.check_labels.append(lbl)
        self.lbl_detections = customtkinter.CTkLabel(
            tab_blocks, text="", anchor="w", justify="left", font=("Helvetica", 13), text_color=DIM_COLOR)
        self.lbl_detections.grid(row=4, column=0, padx=10, pady=(10, 0), sticky="w")

        # ---- Tab Flip / Rotation: tes langsung ----
        customtkinter.CTkLabel(
            tab_motion, anchor="w", justify="left", wraplength=370, text_color=DIM_COLOR,
            font=("Helvetica", 12),
            text="Move blocks like a player would: rotate 90 deg, flip to another face, or lift and put back. "
                 "Each action should show up below. (Lift* = block lifted but top face unchanged; may also be "
                 "a flip to an identical face.)"
        ).grid(row=0, column=0, padx=10, pady=(0, 6), sticky="w")
        self.lbl_counters = customtkinter.CTkLabel(tab_motion, text="", anchor="w", font=("Helvetica", 16, "bold"))
        self.lbl_counters.grid(row=1, column=0, padx=10, pady=2, sticky="w")
        self.lbl_tracked = customtkinter.CTkLabel(tab_motion, text="", anchor="w", font=("Helvetica", 13))
        self.lbl_tracked.grid(row=2, column=0, padx=10, pady=2, sticky="w")
        self.lbl_events = customtkinter.CTkLabel(
            tab_motion, text="", anchor="w", justify="left", font=("Helvetica", 13), text_color=DIM_COLOR)
        self.lbl_events.grid(row=3, column=0, padx=10, pady=(6, 6), sticky="w")
        customtkinter.CTkButton(tab_motion, text="Reset counters", fg_color="gray30", hover_color="gray40",
                                command=self.reset_counters).grid(row=4, column=0, padx=10, pady=4, sticky="w")

        # ---- Tab Tuning ----
        self.lbl_conf, self.sl_conf = self._slider_row(
            tab_tuning, 0, 0.30, 0.95, 13, self._on_conf,
            "Higher = fewer false blocks, but a real block may be missed.")
        self.lbl_stable, self.sl_stable = self._slider_row(
            tab_tuning, 1, 2, 15, 13, self._on_stable,
            "Frames a face must stay the same before a rotation/flip counts. Higher = fewer false counts, slower response.")
        self.lbl_lift, self.sl_lift = self._slider_row(
            tab_tuning, 2, 5, 40, 35, self._on_lift,
            "Box size change treated as 'lifted'. Lower = more sensitive.")
        customtkinter.CTkButton(tab_tuning, text="Reset defaults", fg_color="gray30", hover_color="gray40",
                                command=self.reset_tuning).grid(row=9, column=0, padx=10, pady=(8, 0), sticky="w")

        customtkinter.CTkButton(self, text="Save & Close", command=self.close).grid(
            row=3, column=0, padx=20, pady=(0, 15), sticky="ew")

        self._sync_tuning()
        self._refresh()
        self.after(200, self.lift)

    # ---- tuning ----
    def _slider_row(self, parent, idx, lo, hi, steps, command, hint):
        lbl = customtkinter.CTkLabel(parent, text="", anchor="w", font=("Helvetica", 15))
        lbl.grid(row=idx * 3, column=0, padx=10, pady=(4, 0), sticky="w")
        sl = customtkinter.CTkSlider(parent, from_=lo, to=hi, number_of_steps=steps, command=command, width=360)
        sl.grid(row=idx * 3 + 1, column=0, padx=10, pady=(0, 0), sticky="w")
        customtkinter.CTkLabel(parent, text=hint, anchor="w", justify="left", wraplength=370,
                               text_color=DIM_COLOR, font=("Helvetica", 12)).grid(
            row=idx * 3 + 2, column=0, padx=10, pady=(0, 4), sticky="w")
        return lbl, sl

    def _sync_tuning(self):
        s = camera_settings
        self.sl_conf.set(s["conf"])
        self.sl_stable.set(s["stable_frames"])
        self.sl_lift.set(round(s["lift_ratio"] * 100))
        self.lbl_conf.configure(text=f"Detection confidence: {s['conf']:.2f}")
        self.lbl_stable.configure(text=f"Stable frames: {s['stable_frames']}")
        self.lbl_lift.configure(text=f"Lift sensitivity: {round(s['lift_ratio'] * 100)}%")

    def _on_conf(self, value):
        camera_settings["conf"] = round(float(value), 2)
        self.lbl_conf.configure(text=f"Detection confidence: {camera_settings['conf']:.2f}")

    def _on_stable(self, value):
        camera_settings["stable_frames"] = int(round(value))
        self.lbl_stable.configure(text=f"Stable frames: {camera_settings['stable_frames']}")

    def _on_lift(self, value):
        camera_settings["lift_ratio"] = round(int(round(value)) / 100, 2)
        self.lbl_lift.configure(text=f"Lift sensitivity: {round(camera_settings['lift_ratio'] * 100)}%")

    def reset_tuning(self):
        camera_settings.update(DETECT_DEFAULTS)
        self._sync_tuning()

    def reset_counters(self):
        self.app.check_tracker = BlockRotationTracker(
            stable_frames=camera_settings["stable_frames"], lift_area_ratio=camera_settings["lift_ratio"])

    # ---- refresh langsung ----
    def _refresh(self):
        try:
            if not self.winfo_exists():
                return
            self._update_preview()
            self._update_blocks_tab()
            self._update_motion_tab()
            self._after_id = self.after(100, self._refresh)
        except Exception:
            pass  # dialog sedang ditutup

    def _update_preview(self):
        frame = self.app.cam_preview_frame
        if frame is None:
            return
        h, w = frame.shape[:2]
        scale = min(self.PREVIEW_W / w, self.PREVIEW_H / h)
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = Image.fromarray(frame).resize(size, Image.Resampling.BILINEAR)
        self._preview_img = customtkinter.CTkImage(light_image=img, size=size)
        self.preview.configure(image=self._preview_img)

    def _update_blocks_tab(self):
        blocks = list(self.app.cam_blocks)
        found = len(blocks)
        read = sum(1 for b in blocks if b[2] != 0)
        square = blocks_form_square([(b[0], b[1]) for b in blocks])
        mp_ok = self.app.mp_hands_detector is not None
        hand_seen = bool(self.app.hand_points)

        checks = [
            (found == 4, f"Blocks detected: {found}/4"),
            (found > 0 and read == found, f"Faces readable: {read}/{found}"),
            (square, "Blocks form a square" if found == 4 else "Blocks form a square (need 4 blocks)"),
        ]
        for lbl, (ok, text) in zip(self.check_labels[:3], checks):
            lbl.configure(text=("[OK]  " if ok else "[ ! ]  ") + text, text_color=OK_COLOR if ok else BAD_COLOR)
        if not mp_ok:
            self.check_labels[3].configure(text="[ - ]  Hand tracking unavailable (lift detection limited)",
                                           text_color=DIM_COLOR)
        else:
            self.check_labels[3].configure(
                text="[OK]  Hand tracking: hand in view" if hand_seen else "[ - ]  Hand tracking: no hand in view",
                text_color=OK_COLOR if hand_seen else DIM_COLOR)

        if all(ok for ok, _ in checks):
            self.verdict.configure(text="READY: blocks detected and readable", text_color=OK_COLOR)
        else:
            reason = next(t for ok, t in checks if not ok)
            self.verdict.configure(text=f"NOT READY: {reason}", text_color=BAD_COLOR)

        lines = []
        for i, (cx, cy, state, *_rest) in enumerate(sorted(blocks, key=lambda b: (b[0], b[1]))):
            conf = _rest[-1] if _rest else 0.0
            note = "" if state else "   <- check lighting / brightness / hand"
            lines.append(f"#{i + 1} at ({cx},{cy})  {FACE_NAMES.get(state, '?')}  conf {conf:.2f}{note}")
        self.lbl_detections.configure(text="\n".join(lines) if lines else "No blocks detected.")

    def _update_motion_tab(self):
        tr = self.app.check_tracker
        if tr is None:
            return
        tot = tr.summary()
        self.lbl_counters.configure(
            text=f"Rotations: {tot['rotations']} ({tot['degrees']} deg)   Flips: {tot['flips']}   Lifts*: {tot['hidden']}")
        tracked = []
        for tid, t in sorted(tr.tracks.items()):
            name = FACE_NAMES.get(t["state"], "...") if t["state"] else "..."
            tracked.append(f"#{tid} {name}" + (" (in hand)" if t["ep"] is not None else ""))
        self.lbl_tracked.configure(text="Tracked: " + ("  |  ".join(tracked) if tracked else "none"))
        recent = [describe_tracker_event(ev) for ev in reversed(tr.events[-6:])]
        self.lbl_events.configure(text="\n".join(recent) if recent else "No events yet.")

    def close(self):
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        save_camera_settings()
        self.app.check_dialog = None
        self.app.check_tracker = None
        self.destroy()

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Rotation Counter


class TimeIn(customtkinter.CTk):

    def report_callback_exception(self, exc, val, tb):
        """Capture tkinter callback exceptions with full traceback."""
        log_exception("Tk callback exception (TimeIn)", exc, val, tb)

    def __init__(self):

        super().__init__()
        self.title("Block Design Test")

        # Set window size based on display mode
        if DISPLAY_HALF:
            self.geometry("960x560")
        else:
            self.geometry("1200x800")  # Larger window for full display mode
        
        # Configure main grid with different weights based on display mode
        if DISPLAY_HALF:
            # Half display mode (4:1 ratio for top:bottom)
            self.grid_rowconfigure(0, weight=4)  # Top part (larger portion)
            self.grid_rowconfigure(1, weight=1)  # Bottom part (smaller portion)
            self.grid_columnconfigure(0, weight=1)
            self.grid_columnconfigure(1, weight=1)
            
            # Create a container for the top half
            self.top_container = customtkinter.CTkFrame(self)
            self.top_container.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 150))
            self.top_container.grid_columnconfigure(0, weight=1)
            self.top_container.grid_columnconfigure(1, weight=1)
            self.top_container.grid_rowconfigure(0, weight=1)
            self.top_container.grid_rowconfigure(1, weight=0)  # For the button row
            
            # Content container for half display
            self.content_container = customtkinter.CTkFrame(self.top_container)
            self.content_container.grid(row=0, column=0, columnspan=2, sticky="nsew")
            self.content_container.grid_columnconfigure(0, weight=1)
            self.content_container.grid_columnconfigure(1, weight=1)
            self.content_container.grid_rowconfigure(0, weight=1)
        else:
            # Full display mode (use full window)
            self.grid_rowconfigure(0, weight=1)
            self.grid_columnconfigure(0, weight=1)
            self.grid_columnconfigure(1, weight=1)
            
            # Main container for full display
            self.top_container = customtkinter.CTkFrame(self)
            self.top_container.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
            self.top_container.grid_columnconfigure(0, weight=1)
            self.top_container.grid_columnconfigure(1, weight=1)
            self.top_container.grid_rowconfigure(0, weight=1)
            
            # Content container for full display
            self.content_container = customtkinter.CTkFrame(self.top_container)
            self.content_container.grid(row=0, column=0, columnspan=2, sticky="nsew")
            self.content_container.grid_columnconfigure(0, weight=1)
            self.content_container.grid_columnconfigure(1, weight=1)
            self.content_container.grid_rowconfigure(0, weight=1)

        # --------------------------------------------------- Design Section

        # Left side container for timer and design
        self.left_side = customtkinter.CTkFrame(self.content_container)
        
        # If camera is hidden, center the left_side by using columnspan
        if HIDE_CAMERA:
            # Center the design section by spanning both columns
            if DISPLAY_HALF:
                self.left_side.grid(row=0, column=0, columnspan=2, sticky="", padx=10, pady=10)
            else:
                self.left_side.grid(row=0, column=0, columnspan=2, sticky="", padx=20, pady=20)
        else:
            # Normal layout with camera on the right
            if DISPLAY_HALF:
                self.left_side.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
            else:
                self.left_side.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
            
        self.left_side.grid_rowconfigure(1, weight=1)
        self.left_side.grid_columnconfigure(1, weight=1)

        # Timer frame on the left side
        timer_width = 120 if DISPLAY_HALF else 150
        timer_font_size = 30 if DISPLAY_HALF else 36
        
        self.timer_frame = customtkinter.CTkFrame(self.left_side, corner_radius=6, fg_color="gray30", width=timer_width)
        self.timer_frame.grid(row=0, column=0, rowspan=2, padx=(0, 10), pady=10, sticky="ns")
        self.timer_frame.pack_propagate(False)
        
        # Timer label with larger font
        self.timer_label = customtkinter.CTkLabel(
            self.timer_frame, 
            text="00:00.00", 
            font=("Helvetica", timer_font_size, "bold"),
            text_color="white"
        )
        self.timer_label.pack(expand=True, fill="both")
        self.timer_running = False
        self.start_time = 0

        # Configure column weights for left_side (block design) and content_container (camera)
        self.left_side.grid_columnconfigure(1, weight=1)  # Block design column
        
        if not HIDE_CAMERA:
            self.content_container.grid_columnconfigure(1, weight=3)  # Camera column (3x wider than block design)
        else:
            # When camera is hidden, make both columns equal weight for centering
            self.content_container.grid_columnconfigure(0, weight=1)
            self.content_container.grid_columnconfigure(1, weight=1)

        # Design frames next to the timer
        self.design_frame_0 = TextFrame(self.left_side, "Block Design")
        self.design_frame_0.grid(row=0, column=1, padx=0, pady=(0, 5), sticky="nsew")

        self.design_frame_1 = ImageFrame(self.left_side, "")
        self.design_frame_1.grid(row=1, column=1, padx=0, pady=0, sticky="nsew")
        self.design_frame_1.grid_propagate(False)

        # --------------------------------------------------- Camera Section

        # Only create camera section if HIDE_CAMERA is False
        if not HIDE_CAMERA:
            camera_padx = 10 if DISPLAY_HALF else 20
            camera_pady = (40, 10) if DISPLAY_HALF else (50, 20)
            
            self.video_frame_0 = TextFrame(self.content_container, "Upper Table Camera")
            self.video_frame_0.grid(row=0, column=1, padx=camera_padx, pady=0, sticky="nsew")
            
            self.video_frame_1 = ImageFrame(self.content_container, "")
            self.video_frame_1.grid(row=0, column=1, padx=camera_padx, pady=camera_pady, sticky="nsew")
            self.video_frame_1.grid_propagate(False)
        else:
            # Camera hidden - set to None
            self.video_frame_0 = None
            self.video_frame_1 = None
            print(">>> HIDE_CAMERA enabled - Camera section hidden")
        
        # Adjust image sizes to fit the frames
        if DISPLAY_HALF:
            self.frame_width = 400
            self.frame_height = 400
        else:
            self.frame_width = 600
            self.frame_height = 600
        
        # Load and resize the initial image
        IMAGE_PATH = os.path.join(BASE_DIR, 'FILES', 'TEST_1000x1000', '0Bx.jpg')
        self.image = customtkinter.CTkImage(
            light_image=Image.open(IMAGE_PATH).resize((self.frame_width, self.frame_height), Image.Resampling.LANCZOS),
            size=(self.frame_width, self.frame_height)
        )
        # Create the label once and store it as an instance variable
        self.image_label = customtkinter.CTkLabel(master=self.design_frame_1, text='')
        self.image_label.pack(expand=True, fill="both")
        self.image_label.configure(image=self.image)

        # Camera label - Configure grid for video_frame_1 (only if camera not hidden)
        if not HIDE_CAMERA:
            self.video_frame_1.grid_rowconfigure(0, weight=1)
            self.video_frame_1.grid_columnconfigure(0, weight=1)
            
            # Create camera label with grid
            self.camera = customtkinter.CTkLabel(self.video_frame_1, text="", anchor="center")
            self.camera.grid(row=0, column=0, sticky="nsew")
        else:
            # Camera hidden - set to None
            self.camera = None

        # Add the start button
        button_font_size = 30 if DISPLAY_HALF else 36
        button_pady = 10 if DISPLAY_HALF else 20
        
        self.button_0 = customtkinter.CTkButton(
            self.top_container, 
            text="(START)", 
            font=("Helvetica", button_font_size), 
            command=self.button_0_callback
        )
        if DISPLAY_HALF:
            self.button_0.grid(row=1, column=0, columnspan=2, padx=20, pady=button_pady, sticky="ew")
        else:
            # In full mode, place the button at the bottom of the left panel
            self.button_0.grid(row=2, column=0, columnspan=2, padx=20, pady=button_pady, sticky="ew")

        # Camera settings button (hanya tampil sebelum tes dimulai)
        self.button_cam = customtkinter.CTkButton(
            self.top_container, text="Camera Settings", width=140, height=32,
            fg_color="gray30", hover_color="gray40", command=self.open_camera_settings
        )
        self.button_cam.place(relx=1.0, x=-12, y=8, anchor="ne")
        self.button_cam.lift()
        self.button_check = customtkinter.CTkButton(
            self.top_container, text="Detection Check", width=140, height=32,
            fg_color="gray30", hover_color="gray40", command=self.open_detection_check
        )
        self.button_check.place(relx=1.0, x=-12, y=46, anchor="ne")
        self.button_check.lift()

        # Bottom container (only used in half display mode)
        if DISPLAY_HALF:
            self.bottom_container = customtkinter.CTkFrame(self)
            self.bottom_container.grid(row=1, column=0, columnspan=2, sticky="nsew")

        #self.start_zero = time.time()

        self.timer_task_all = []

        self.level_times = {n: 0 for n in range(1, LEVEL_COUNT + 1)}
        self.task_flag   = {n: True for n in range(1, LEVEL_COUNT + 1)}

        self.current_question = 1

        self.cognitive_age_list = []

        self.nick_name = nick_name
        self.gender_code = gender_code
        self.age_range_code = age_range_code

        self.retry_button = None

        #self.task_state = 1

        # Set the maximum level from environment variable
        self.max_level = MAX_LEVEL
        
        # Bind Enter key to skip level
        self.bind('<Return>', self.skip_current_level)
        
        # Initialize YOLO detection thread
        self.yolo_thread = YOLODetectionThread(model_yolo, USE_BANTAL_MODEL)
        self.yolo_thread.start()
        self.latest_detections = []
        self.hand_points = []   # landmark tangan (px) dari MediaPipe, frame terakhir
        self.cam_dialog = None
        self.cam_preview_frame = None   # frame terakhir (dengan overlay) untuk preview dialog
        self.cam_stats = (0, 0)         # (balok terdeteksi, sisi terbaca)
        self.cam_blocks = []            # observasi balok frame terakhir untuk dialog check
        self.check_dialog = None
        self.check_tracker = None       # tracker terpisah untuk tes flip/rotasi sebelum START
        self.rotation_tracker = BlockRotationTracker(
            stable_frames=int(os.getenv('ROTATION_STABLE_FRAMES', '6')))
        self.frame_count = 0
        self.yolo_skip_frames = int(os.getenv('YOLO_SKIP_FRAMES', '2'))  # Process every 3rd frame
        
        # Initialize MediaPipe Hands once (not every frame)
        self.mp_hands_detector = None
        if mp_hands is not None:
            try:
                self.mp_hands_detector = mp_hands.Hands(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    max_num_hands=2
                )
            except Exception as e:
                print(f">>> Failed to initialize MediaPipe Hands: {e}. Hand landmark detection disabled.")
        
        # FPS monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Skip MediaPipe on same frames as YOLO for better performance
        self.mediapipe_skip_frames = int(os.getenv('MEDIAPIPE_SKIP_FRAMES', '2'))
        
        # Debug mode
        self.debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        
        # Button mode for image display control
        self.button_mode = BUTTON_MODE
        self.image_visible = False
        self.image_show_time = None
        self.image_display_duration = 5.0  # 5 seconds
        
        # Serial communication thread for button mode
        self.serial_thread = None
        if self.button_mode:
            print(">>> BUTTON_MODE enabled - Image will show for 5 seconds then hide")
            self.serial_thread = SerialReaderThread()
            self.serial_thread.start()
        
        # Cache for level images
        self.cached_level_images = {}
        self.et = EyeTrackerHandler(
            on_gaze=self._on_gaze_update,
            on_status_change=lambda s: print(f"[EyeTracker] {s}")
        )
        self.et.connect()

        self._gaze_dot_size = 30
        self.gaze_dot = customtkinter.CTkCanvas(
            self, width=self._gaze_dot_size, height=self._gaze_dot_size,
            highlightthickness=0, bg="#000001"  # warna unik, dipakai sbg transparent key
        )
        # TIDAK perlu .place() atau .lift()/.lower() di sini saat ini
        self._latest_gaze = None
        self._screen_w = self.winfo_screenwidth()
        self._screen_h = self.winfo_screenheight()
        self.aoi = AoiTracker(self._screen_w, self._screen_h)
        self.after(16, self._poll_gaze)
        self.after(100, self._refresh_aoi_rects)
        self.game_started = False
        self.preload_level_images()
        self.after(200, self.streaming)

    def preload_level_images(self):
        """Preload and cache all level images to avoid I/O lag during gameplay"""
        print(">>> Preloading level images...")
        for variant, path in LEVEL_PATHS.items():
            try:
                img = Image.open(path)
                # Cache resized version for display
                self.cached_level_images[variant] = img.resize(
                    (self.frame_width, self.frame_height), 
                    Image.Resampling.LANCZOS
                )
            except Exception as e:
                print(f"Error loading {variant}: {e}")
        print(f">>> Cached {len(self.cached_level_images)} level images")
    
    def get_cached_level_image(self, variant):
        """Get cached level image or load if not cached"""
        if variant in self.cached_level_images:
            return self.cached_level_images[variant]
        else:
            # Fallback: load on demand
            try:
                img = Image.open(LEVEL_PATHS[variant])
                return img.resize((self.frame_width, self.frame_height), Image.Resampling.LANCZOS)
            except Exception as e:
                print(f"Error loading {variant}: {e}")
                return None

    def end_test(self):
        self.current_question = 9
        self.reset_timer()
        self.stop_timer()
        self.current_level_button.grid_remove()
        total_gaze_duration = self.et.gaze_duration_seconds
        aoi = self.aoi.snapshot()
        self.et.stop()
        self.game_started = False
        self.gaze_dot.delete("gaze")
        self.gaze_dot.place_forget()
        self.timer_task_avg = format(float(sum(self.timer_task_all) / len(self.timer_task_all)), ".3f")
        total_time = sum(self.timer_task_all)
        age_real = int(self.age_range_code)
        age_cog = average_cognitive_age(self.cognitive_age_list)

        visuo_spatial = fitness_from_ages(age_cog, age_real)

        print("Your name is " + self.nick_name)
        print("Your gender is " + self.gender_code)
        print("Your Average Completed Time is " + str(self.timer_task_avg) + " second")
        print("Your Current Age is " + str(age_real) + " years")
        print("Your Cognitive Age is " + str(age_cog) + " years")
        print("Your Cognitive Fitness is " + str(visuo_spatial) + " %")

        rot_total = self.rotation_tracker.summary()
        for lvl in sorted(self.rotation_tracker.stats):
            r, f, hd = self.rotation_tracker.level_counts(lvl)
            print(f"Level {lvl}: {r} rotation(s), {f} flip(s), {hd} lift(s) without visible change")
        print(f"Total rotation: {rot_total['rotations']} ({rot_total['degrees']} degrees), "
              f"total flip: {rot_total['flips']}, total lift without visible change: {rot_total['hidden']}")

        play_audio("AUDIO/selesai.wav")

        # Generate SQL file for cronjob to execute
        try:
            import datetime
            
            # Create SQL directory if it doesn't exist
            sql_dir = Path('./cron/sql')
            sql_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename with timestamp
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            sql_filename = sql_dir / f'participant_{timestamp}.sql'
            
            # Prepare values
            name = self.nick_name.replace("'", "''")  # Escape single quotes
            real_age = int(self.age_range_code)
            gender = self.gender_code
            estimated_age = int(age_cog)
            cognitive_fitness = f"{visuo_spatial}%"
            avg_time_all = float(self.timer_task_avg)
            avg_time_1 = float(self.timer_task_all[0]) if len(self.timer_task_all) > 0 else 'NULL'
            avg_time_2 = float(self.timer_task_all[1]) if len(self.timer_task_all) > 1 else 'NULL'
            avg_time_3 = float(self.timer_task_all[2]) if len(self.timer_task_all) > 2 else 'NULL'
            avg_time_4 = float(self.timer_task_all[3]) if len(self.timer_task_all) > 3 else 'NULL'
            avg_time_5 = float(self.timer_task_all[4]) if len(self.timer_task_all) > 4 else 'NULL'
            avg_time_6 = float(self.timer_task_all[5]) if len(self.timer_task_all) > 5 else 'NULL'
            avg_time_7 = float(self.timer_task_all[6]) if len(self.timer_task_all) > 6 else 'NULL'
            avg_time_8 = float(self.timer_task_all[7]) if len(self.timer_task_all) > 7 else 'NULL'
            
            # Build SQL content
            sql_content = f"""-- Auto-generated SQL for participant data
-- Generated at: {datetime.datetime.now().isoformat()}

DO $$
DECLARE
    v_participant_id INTEGER;
    v_timestamp TIMESTAMP;
BEGIN
    -- Check if participant exists
    SELECT id INTO v_participant_id
    FROM bdt.participant_data
    WHERE LOWER(name) = LOWER('{name}')
    AND real_age = {real_age}
    AND gender = '{gender}'
    LIMIT 1;

    IF v_participant_id IS NOT NULL THEN
        -- Update existing participant
        UPDATE bdt.participant_data
        SET 
            estimated_cognitive_age = {estimated_age},
            cognitive_fitness = '{cognitive_fitness}',
            avg_time_all = {avg_time_all},
            avg_time_1 = {avg_time_1},
            avg_time_2 = {avg_time_2},
            avg_time_3 = {avg_time_3},
            avg_time_4 = {avg_time_4},
            avg_time_5 = {avg_time_5},
            avg_time_6 = {avg_time_6},
            avg_time_7 = {avg_time_7},
            avg_time_8 = {avg_time_8},
            updated_at = CURRENT_TIMESTAMP
        WHERE id = v_participant_id
        RETURNING updated_at INTO v_timestamp;

        -- Insert history record
        INSERT INTO bdt.participant_data_history (
            name, real_age, gender, estimated_cognitive_age,
            cognitive_fitness, avg_time_all, avg_time_1,
            avg_time_2, avg_time_3, avg_time_4, avg_time_5,
            avg_time_6, avg_time_7, avg_time_8, created_at
        ) VALUES (
            '{name}', {real_age}, '{gender}', {estimated_age},
            '{cognitive_fitness}', {avg_time_all}, {avg_time_1},
            {avg_time_2}, {avg_time_3}, {avg_time_4}, {avg_time_5},
            {avg_time_6}, {avg_time_7}, {avg_time_8}, v_timestamp
        );

        RAISE NOTICE 'Updated existing participant: %', '{name}';
    ELSE
        -- Insert new participant
        INSERT INTO bdt.participant_data (
            name, real_age, gender, estimated_cognitive_age,
            cognitive_fitness, avg_time_all, avg_time_1,
            avg_time_2, avg_time_3, avg_time_4, avg_time_5,
            avg_time_6, avg_time_7, avg_time_8, created_at
        ) VALUES (
            '{name}', {real_age}, '{gender}', {estimated_age},
            '{cognitive_fitness}', {avg_time_all}, {avg_time_1},
            {avg_time_2}, {avg_time_3}, {avg_time_4}, {avg_time_5},
            {avg_time_6}, {avg_time_7}, {avg_time_8}, CURRENT_TIMESTAMP
        )
        RETURNING created_at INTO v_timestamp;

        -- Insert initial history record
        INSERT INTO bdt.participant_data_history (
            name, real_age, gender, estimated_cognitive_age,
            cognitive_fitness, avg_time_all, avg_time_1,
            avg_time_2, avg_time_3, avg_time_4, avg_time_5,
            avg_time_6, avg_time_7, avg_time_8, created_at
        ) VALUES (
            '{name}', {real_age}, '{gender}', {estimated_age},
            '{cognitive_fitness}', {avg_time_all}, {avg_time_1},
            {avg_time_2}, {avg_time_3}, {avg_time_4}, {avg_time_5},
            {avg_time_6}, {avg_time_7}, {avg_time_8}, v_timestamp
        );

        RAISE NOTICE 'Inserted new participant: %', '{name}';
    END IF;
END $$;
"""
            
            # Write SQL file
            with open(sql_filename, 'w', encoding='utf-8') as f:
                f.write(sql_content)
            
            print(f">>> SQL file generated: {sql_filename}")
            print(">>> Cronjob will execute this file automatically")
            
        except Exception as e:
            print(">>> Error generating SQL file:")
            print(f"Error: {e}")

        # IMAGE_PATH = os.path.join(BASE_DIR, 'FILES', 'TEST_1000x1000', '09.jpg')
        # self.image = customtkinter.CTkImage(light_image=Image.open(IMAGE_PATH), size=(self.frame_width, self.frame_height))
        # self.image_label.configure(image=self.image)
        # self.show_retry_button()
        # result_test = (
            # f"Waktu Penyelesaian: {total_time:.2f} detik\n"
            # f"Durasi Menatap Layar: {total_gaze_duration:.2f} detik\n"
            # f"AOI 1 (Stopwatch): {aoi['AOI1']:.2f} detik   |   "
            # f"AOI 2 (Level): {aoi['AOI2']:.2f} detik\n"
            # f"B1: {aoi['B1']:.2f} detik   B2: {aoi['B2']:.2f} detik   "
            # f"B3: {aoi['B3']:.2f} detik   B4: {aoi['B4']:.2f} detik"
        # )
        data = {
            "name": self.nick_name,
            "total_time": total_time,
            "gaze": total_gaze_duration,
            "aoi": aoi,
            "levels": list(self.timer_task_all),
            "avg": float(self.timer_task_avg),
            "age_real": age_real,
            "age_cog": age_cog,
            "fitness": visuo_spatial,
        }
        # jeda 1,5 dtk (audio "selesai" terdengar, GUI tetap hidup), lalu transisi
        self.after(1500, lambda: self._transition_to_post(data))

    def _fade(self, a0, a1, steps=10, ms=20, done=None):
        def step(i=0):
            self.attributes("-alpha", a0 + (a1 - a0) * i / steps)
            if i < steps:
                self.after(ms, lambda: step(i + 1))
            elif done:
                done()
        step()

    def _transition_to_post(self, data):
        def swap():
            self.build_post_game(data)
            self.update_idletasks()
            self._fade(0.0, 1.0)
        self._fade(1.0, 0.0, done=swap)

    def _card(self, parent, title, value, big, mid):
        f = customtkinter.CTkFrame(parent, corner_radius=12)
        customtkinter.CTkLabel(f, text=title, font=("Helvetica", mid)).pack(pady=(12, 0))
        customtkinter.CTkLabel(f, text=value, font=("Helvetica", big, "bold")).pack(expand=True)
        return f

    def build_post_game(self, d):
        self.content_container.grid_remove()
        pf = customtkinter.CTkFrame(self.top_container)
        pf.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.post_frame = pf
        pf.grid_columnconfigure((0, 1), weight=1, uniform="c")
        for r, w in ((0, 0), (1, 2), (2, 5), (3, 0)):
            pf.grid_rowconfigure(r, weight=w)

        H = max(self.winfo_height(), 600)
        big, mid = int(H * 0.05), int(H * 0.022)
        aoi = d["aoi"]
        total = max(d["total_time"], 0.001)               # pembagi: waktu tes (sebelumnya d["gaze"])
        screen_t = aoi["AOI1"] + aoi["AOI2"]
        block_t = max(d["total_time"] - screen_t, 0)      # = melihat balok fisik

        customtkinter.CTkLabel(pf, text=f"Tes Selesai — {d['name']}",
            font=("Helvetica", int(H * 0.045), "bold")
        ).grid(row=0, column=0, columnspan=2, pady=(20, 10))

        cards = customtkinter.CTkFrame(pf, fg_color="transparent")
        cards.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=20)
        cards.grid_rowconfigure(0, weight=1)
        items = [("Waktu Penyelesaian", f"{d['total_time']:.2f} dtk"),
                 ("Menatap Layar", f"{screen_t:.2f} dtk"),
                 ("Melihat Balok", f"{block_t:.2f} dtk"),
                 ("Rata-rata per Level", f"{d['avg']:.2f} dtk"),
                 ("Usia Kognitif", f"{d['age_cog']} th")]
        for i, (t, v) in enumerate(items):
            cards.grid_columnconfigure(i, weight=1, uniform="k")
            self._card(cards, t, v, big, mid).grid(row=0, column=i, sticky="nsew", padx=8, pady=8)

        left = customtkinter.CTkFrame(pf, corner_radius=12)
        left.grid(row=2, column=0, sticky="nsew", padx=(20, 10), pady=10)
        customtkinter.CTkLabel(left, text="Area Tatapan (AOI)",
            font=("Helvetica", mid, "bold")).pack(pady=(15, 5))
        for name, val in (("AOI 1 · Stopwatch", aoi["AOI1"]),
                          ("AOI 2 · Soal", aoi["AOI2"]),
                          ("Melihat balok fisik", block_t)):
            pct = val / total
            customtkinter.CTkLabel(left, text=f"{name}: {val:.2f} dtk ({pct*100:.0f}%)",
                font=("Helvetica", mid)).pack(pady=(15, 2))
            bar = customtkinter.CTkProgressBar(left, height=int(H * 0.03))
            bar.pack(fill="x", padx=30)
            bar.set(min(pct, 1.0))

        right = customtkinter.CTkFrame(pf, corner_radius=12)
        right.grid(row=2, column=1, sticky="nsew", padx=(10, 20), pady=10)
        right.grid_rowconfigure((0, 1), weight=1, uniform="g")
        right.grid_columnconfigure((0, 1), weight=1, uniform="g")
        mx = max(max(aoi[k] for k in ("B1", "B2", "B3", "B4")), 0.001)
        for key, (r, c) in {"B1": (0, 0), "B2": (0, 1), "B3": (1, 0), "B4": (1, 1)}.items():
            v = int(255 * (1 - aoi[key] / mx))
            cell = customtkinter.CTkFrame(right, fg_color=f"#ff{v:02x}{v:02x}", corner_radius=8)
            cell.grid(row=r, column=c, sticky="nsew", padx=6, pady=6)
            customtkinter.CTkLabel(cell,
                text=f"{key}\n{aoi[key]:.2f} dtk\n{aoi[key]/total*100:.0f}%",
                font=("Helvetica", mid, "bold"), text_color="black").pack(expand=True)

        customtkinter.CTkButton(pf, text="Retry Test", font=("Helvetica", int(H * 0.035)),
            height=int(H * 0.08), command=self.retry_test
        ).grid(row=3, column=0, columnspan=2, sticky="ew", padx=20, pady=(5, 20))


    def _play_praise(self, seconds):
        t = int(seconds)
        for limit, name in ((10, "menakjubkan"), (15, "hebat_sekali"), (20, "mantap"),
                            (25, "kerja_bagus"), (30, "ayo_semangat")):
            if t < limit:
                play_audio(f"AUDIO/{name}.wav")
                return
        play_audio("AUDIO/jangan_menyerah.wav")

    def _record_level(self, level, seconds):
        self.level_times[level] = seconds
        self.timer_task_all.append(seconds)
        self.cognitive_age_list.append(estimate_cognitive_age(seconds))
        self.task_flag[level] = False

    def _start_level(self, level):
        self.current_question = level
        variant = self.get_random_variant(level)
        self.current_variant = variant
        self.load_level_image(variant)
        self.current_level_button.grid_remove()
        self.show_current_level_button(level)
        self.start_task = time.time()
        self.reset_timer()
        self.start_timer()

    def _on_level_solved(self, level):
        if not self.task_flag[level]:
            return
        seconds = round(time.time() - self.start_task, 2)   # timer_return selalu 0
        self._record_level(level, seconds)
        print(f"TASK {level} COMPLETED in {seconds} seconds")
        self._play_praise(seconds)
        if level >= min(self.max_level, LEVEL_COUNT):
            self.end_test()
        else:
            play_audio(f"AUDIO/lanjut_lvl{level + 1}.wav")
            self._start_level(level + 1)

    def _on_level_skipped(self, level, elapsed):
        self._record_level(level, elapsed)
        print(f"TASK {level} SKIPPED after {elapsed} seconds")
        self._play_praise(elapsed)
        if self.max_level != level:
            play_audio(f"AUDIO/lanjut_lvl{level + 1}.wav")
        self.current_question = level + 1
    
    def skip_current_level(self, event=None):
        """Skip the current level when Enter key is pressed"""
        if not hasattr(self, 'current_question') or not hasattr(self, 'timer_running') or not self.timer_running:
            return

        level = self.current_question
        if level < 1 or level > LEVEL_COUNT:
            return

        time_elapsed = round(time.time() - self.start_task, 2)
        print(f"Skipping level {level} after {time_elapsed} seconds")

        self._on_level_skipped(level, time_elapsed)

        # Lanjut ke level berikutnya atau akhiri tes
        if self.current_question <= min(self.max_level, LEVEL_COUNT):
            self._start_level(self.current_question)
        else:
            self.end_test()
        
    
    def get_random_variant(self, level):
        """Get a variant for the specified level, using custom levels if available."""
        # If we have a custom level defined, use it
        if level in CUSTOM_LEVELS:
            return CUSTOM_LEVELS[level]
            
        # Otherwise, choose a random variant (a-d)
        variants = ['a', 'b', 'c', 'd']
        return f"{level}{random.choice(variants)}"

    def open_detection_check(self):
        if self.game_started:
            return
        if self.check_dialog is not None and self.check_dialog.winfo_exists():
            self.check_dialog.lift()
            return
        self.check_tracker = BlockRotationTracker(
            stable_frames=camera_settings["stable_frames"], lift_area_ratio=camera_settings["lift_ratio"])
        self.check_dialog = DetectionCheckDialog(self)

    def open_camera_settings(self):
        if self.game_started:
            return
        if self.cam_dialog is not None and self.cam_dialog.winfo_exists():
            self.cam_dialog.lift()
            return
        self.cam_dialog = CameraSettingsDialog(self)

    def run_start_countdown(self, seconds):
        """Hitung mundur di tengah layar: 3, 2, 1, Mulai! Memakai after() sehingga streaming kamera tetap berjalan."""
        box = customtkinter.CTkFrame(self, width=620, height=340, corner_radius=20)
        box.pack_propagate(False)
        caption = customtkinter.CTkLabel(box, text="Permainan akan dimulai dalam", font=("Helvetica", 28))
        caption.pack(pady=(40, 0))
        number = customtkinter.CTkLabel(box, text="", font=("Helvetica", 150, "bold"))
        number.pack(expand=True)
        box.place(relx=0.5, rely=0.5, anchor="center")
        box.lift()
        self.countdown_box = box

        def finish():
            try:
                box.destroy()
            except Exception:
                pass
            self.countdown_box = None
            self._begin_test()

        def tick(n):
            if n > 0:
                number.configure(text=str(n))
                self.after(1000, lambda: tick(n - 1))
            else:
                caption.configure(text="")
                number.configure(text="Mulai!")
                self.after(700, finish)

        tick(int(seconds))

    def build_game_layout(self):
        # lepas ruang terbuang
        self.top_container.grid_configure(pady=0)
        if DISPLAY_HALF:
            self.bottom_container.grid_remove()
            self.grid_rowconfigure(0, weight=1)
            self.grid_rowconfigure(1, weight=0)

        # sembunyikan layout pra-game + kamera
        self.left_side.grid_remove()
        if self.video_frame_0 is not None:
            self.video_frame_0.grid_remove()
            self.video_frame_1.grid_remove()

        # frame in-game: 2 kolom sama lebar
        self.game_frame = customtkinter.CTkFrame(self.content_container, fg_color="white")
        self.game_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.game_frame.grid_columnconfigure(0, weight=1, uniform="half")
        self.game_frame.grid_columnconfigure(1, weight=1, uniform="half")
        self.game_frame.grid_rowconfigure(0, weight=1)

        # AOI 1 (kiri)
        self.aoi1_frame = customtkinter.CTkFrame(self.game_frame, fg_color="white")
        self.aoi1_frame.grid(row=0, column=0, sticky="nsew")
        self.aoi1_frame.grid_rowconfigure(0, weight=1)
        self.aoi1_frame.grid_columnconfigure(0, weight=1)

        # AOI 2 (kanan) + frame dalam untuk label Level dan gambar soal
        self.aoi2_frame = customtkinter.CTkFrame(self.game_frame, fg_color="white")
        self.aoi2_frame.grid(row=0, column=1, sticky="nsew")
        self.aoi2_frame.grid_columnconfigure(0, weight=1)
        self.aoi2_frame.grid_rowconfigure(0, weight=1)

        self.right_inner = customtkinter.CTkFrame(self.aoi2_frame, fg_color="transparent")
        self.right_inner.grid(row=0, column=0)

        # hitung ukuran dari ruang yang tersedia (frame masih kosong di titik ini)
        self.update_idletasks()
        w2, h2 = self.aoi2_frame.winfo_width(), self.aoi2_frame.winfo_height()
        size = max(int(min(w2 * 0.9, (h2 - 90) * 0.95)), 200)
        self.frame_width = self.frame_height = size

        self.cached_level_images.clear()
        self.preload_level_images()

        w1, h1 = self.aoi1_frame.winfo_width(), self.aoi1_frame.winfo_height()
        tw = max(int(min(w1 * 0.8, h1 * 0.6 * 4 / 3)), 200)
        th = int(tw * 3 / 4)

        # widget yang bergantung pada ukuran: dibuat SEKALI
        self.timer_frame_game = customtkinter.CTkFrame(
            self.aoi1_frame, fg_color="gray90", width=tw, height=th, corner_radius=0)
        self.timer_frame_game.grid(row=0, column=0)
        self.timer_frame_game.grid_propagate(False)
        self.timer_frame_game.pack_propagate(False)

        self.timer_label = customtkinter.CTkLabel(
            self.timer_frame_game, text="00:00.00",
            font=("Helvetica", int(tw * 0.17), "bold"), text_color="black")
        self.timer_label.pack(expand=True)

        self.image_label = customtkinter.CTkLabel(
            self.right_inner, text="", width=size, height=size)
        self.image_label.grid(row=1, column=0)

    def _widget_rect(self, w):
        x, y = w.winfo_rootx(), w.winfo_rooty()
        return (x, y, x + w.winfo_width(), y + w.winfo_height())

    def get_aoi_rects(self):
        return build_rects(self._screen_w, self._screen_h, self._widget_rect(self.image_label))

    def _refresh_aoi_rects(self):
        if self.game_started:
            self.aoi.set_rects(self.get_aoi_rects()) 
        self.after(100, self._refresh_aoi_rects)

    def _on_gaze_update(self, gaze):
        self._latest_gaze = gaze
        if not self.game_started:
            return
        self.aoi.on_gaze(gaze["timestamp"], gaze["x"], gaze["y"])

    def _poll_gaze(self):
        # Dipanggil berulang dari main thread lewat self.after() --
        # di sinilah baru boleh menyentuh widget Tkinter.
        if self.game_started and self._latest_gaze is not None:
            self._draw_gaze_point(self._latest_gaze)
        self.after(16, self._poll_gaze)  # ~60x/detik

    def _draw_gaze_point(self, gaze):
        if not self.game_started:
            return
        
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        abs_x = int(gaze["x"] * screen_w)
        abs_y = int(gaze["y"] * screen_h)

        win_x = self.winfo_rootx()
        win_y = self.winfo_rooty()
        px = abs_x - win_x
        py = abs_y - win_y

        
        r = self._gaze_dot_size // 2
        self.gaze_dot.place(x=px - r, y=py - r)
        self.gaze_dot.delete("gaze")
        self.gaze_dot.create_oval(
            2, 2, self._gaze_dot_size - 2, self._gaze_dot_size - 2,
            fill="cyan", outline="black", width=1, tags="gaze"
        )

    def button_0_callback(self):
        # Hide the start button
        self.button_0.grid_remove()
        self.build_game_layout()
        # Camera settings hanya untuk sebelum tes: sembunyikan tombol dan simpan/tutup dialog
        self.button_cam.place_forget()
        self.button_check.place_forget()
        if self.check_dialog is not None:
            self.check_dialog.close()
        if self.cam_dialog is not None:
            self.cam_dialog.close()
        # Show the current level button in the same position
        self.show_current_level_button(self.current_question)
        
        # Hitung mundur di layar (tidak memblokir kamera/GUI); tes dimulai setelah selesai
        self.run_start_countdown(COUNTDOWN_SECONDS)

    def _begin_test(self):
        """Dipanggil setelah hitung mundur selesai: tes benar-benar dimulai."""
        self.aoi.reset()        
        self.game_started = True
        print("START")
        self.et.start()
        #self.start_task = time.time()
        
        # Load and resize the initial image
        variant = self.get_random_variant(self.current_question)
        self.current_variant = variant
        
        # Use helper method to load image with button mode support
        self.load_level_image(variant)
        
        self.start_task = time.time()

        # Reset and start the timer
        self.reset_timer()
        self.start_timer()

    def show_current_level_button(self, level):
        self.current_level_button = customtkinter.CTkLabel(
            self.right_inner, text="Level " + str(level),
            font=("Helvetica", 30, "bold"), text_color="black")
        self.current_level_button.grid(row=0, column=0, pady=(10, 5))
    
    def show_retry_button(self):
        self.retry_button = customtkinter.CTkButton(
            self.top_container,
            text="Retry Test",
            font=("Helvetica", 30),
            command=self.retry_test
        )
        
        # Position the retry button in the same place as the start button was
        self.retry_button.grid(row=1, column=0, columnspan=2, padx=20, pady=10, sticky="ew")

    def retry_test(self):
        # Reset the application state
        self.destroy()  # Close the current window
        # Restart the application
        app_input = App_Input()
        app_input.mainloop()

        if nick_name:  # Only proceed if user provided input
            app = TimeIn()
            app.after(0, lambda:app.state('zoomed'))
            app.mainloop()

    def update_timer(self):
        if self.timer_running:
            elapsed = time.time() - self.start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            milliseconds = int((elapsed % 1) * 100)
            self.timer_label.configure(text=f"{minutes:02d}:{seconds:02d}.{milliseconds:02d}")
            self.after(50, self.update_timer)  # Reduced from 10ms to 50ms

    def start_timer(self):
        if not self.timer_running:
            self.start_time = time.time()
            self.timer_running = True
            self.update_timer()

    def stop_timer(self):
        self.timer_running = False

    def reset_timer(self):
        self.timer_label.configure(text="00:00.00")
        self.timer_running = False

    def load_level_image(self, variant):
        """Load level image with button mode support"""
        # Load cached image
        cached_img = self.get_cached_level_image(variant)
        if cached_img:
            self.image = customtkinter.CTkImage(
                light_image=cached_img,
                size=(self.frame_width, self.frame_height)
            )
        
        # Button mode: Show image for 5 seconds then hide
        if self.button_mode:
            self.image_label.configure(image=self.image)
            self.image_visible = True
            self.image_show_time = time.time()
            print(">>> Image displayed (will hide after 5 seconds)")
        else:
            # Normal mode: Always show image
            self.image_label.configure(image=self.image)
    
    def handle_button_mode(self):
        """Handle button mode image visibility logic"""
        if not self.button_mode:
            return
        
        # Check if image should be hidden after 5 seconds
        if self.image_visible and self.image_show_time:
            elapsed = time.time() - self.image_show_time
            if elapsed >= self.image_display_duration:
                # Hide image by showing blank
                blank_image = Image.new('RGB', (self.frame_width, self.frame_height), color='gray')
                self.image = customtkinter.CTkImage(
                    light_image=blank_image,
                    size=(self.frame_width, self.frame_height)
                )
                self.image_label.configure(image=self.image)
                self.image_visible = False
                self.image_show_time = None
                print(">>> Image hidden (press button to show again)")
        
        # Check for button press from ESP32
        if self.serial_thread:
            message = self.serial_thread.get_message()
            if message == "disable_image" and not self.image_visible:
                # Show image again for 5 seconds
                variant = self.current_variant
                cached_img = self.get_cached_level_image(variant)
                if cached_img:
                    self.image = customtkinter.CTkImage(
                        light_image=cached_img,
                        size=(self.frame_width, self.frame_height)
                    )
                self.image_label.configure(image=self.image)
                self.image_visible = True
                self.image_show_time = time.time()
                print(">>> Button pressed - Image displayed (will hide after 5 seconds)")
    
    # code for video streaming
    def streaming(self):

        if self.game_started:
            self.button_0._state = "disabled"

        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            self.after(100, self.streaming)
            return
            
        frame = apply_camera_settings(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Handle button mode image visibility
        self.handle_button_mode()

        detections = self._poll_yolo(frame)  # YOLO
        track_input, box_num, box_face, box_design, box_distance, pos_x, pos_y = self._scan_blocks(frame, detections)
        self._update_trackers(frame, track_input)  # Update trackers with new detections
        self._check_level(frame, box_design, box_distance, pos_x, pos_y)  # Check if the current level is solved
        self._process_hands(frame)  # Process hand landmarks using MediaPipe Hands
        self.cam_preview_frame = frame

        if self._render_preview(frame):
            return
        self._update_fps() 
        self.after(10, self.streaming)
        
    def _scan_blocks(self, frame, detections):
        # Frame threshold 
        imgBlur = cv2.GaussianBlur(frame, (7,7), 1)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_RGB2GRAY)
        ret, imgThres = cv2.threshold(imgGray, 175, 255, cv2.THRESH_BINARY) 
                
        box_num = 0
        box_face = 0
        box_design = []
        box_distance = []
        
        track_input = []   # (cx, cy, face_state, x1, y1, x2, y2, conf) untuk rotation tracker
        #avg_confidence = []
        
        pos_x = []
        pos_y = []
        
        for x1, y1, x2, y2, conf_pred, cls_id, cls in detections:
        
            if conf_pred > camera_settings["conf"]:
        
                #avg_confidence.append( round(conf_pred,2) )
        
                center_x = int ((x1+x2)/2)
                center_y = int ((y1+y2)/2)
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                w = int (x2-x1)
                h = int (y2-y1)
        
                box_distance.append( int (math.sqrt( pow(center_x, 2) + pow(center_y, 2) )) )
                #print(center_x, center_y)
        
                pos_x.append( int (center_x) )
                pos_y.append( int (center_y) )
                        
                # Optimize: check bounds before resize to avoid errors
                if y1 >= 0 and y2 <= imgThres.shape[0] and x1 >= 0 and x2 <= imgThres.shape[1] and (y2-y1) > 0 and (x2-x1) > 0:
                    dim = (100, 100)
                    imgBox = cv2.resize(imgThres[y1:y2, x1:x2], dim, interpolation = cv2.INTER_AREA)
                else:
                    # Skip invalid box
                    continue
                #cv2.imshow("Box_"+str(box_num), imgBox)
                        
                box_class = [ imgBox[50,25], imgBox[75,50], imgBox[50,75], imgBox[25,50] ]
                face = FACE_STATE_MAP.get(tuple(int(v) for v in box_class), 0)
                track_input.append((center_x, center_y, FACE_STATE_MAP.get(tuple(int(v) for v in box_class), 0), x1, y1, x2, y2, float(conf_pred)))
        
                if face:
                    box_face = face
                    box_design.append(face)     
                        
                cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), (0, 255, 0), 2)
        
                box_num = box_num + 1
                roi_label = frame[y1:y1+50, x1:x1+50]
        
                if box_face in FACE_OVERLAYS:
                    mask, face_img = FACE_OVERLAYS[box_face]
                    try:
                        roi_label[np.where(mask)] = 0
                        roi_label += face_img
                    except IndexError:
                        pass
                    
        return track_input, box_num, box_face, box_design, box_distance, pos_x, pos_y
    
    def _update_trackers(self, frame, track_input):
        self.cam_stats = (len(track_input), sum(1 for o in track_input if o[2] != 0))
        self.cam_blocks = track_input
        
        # Setelan deteksi dari dialog Detection Check berlaku langsung
        self.rotation_tracker.stable_frames = camera_settings["stable_frames"]
        self.rotation_tracker.lift_area_ratio = camera_settings["lift_ratio"]
        if not self.game_started and self.check_dialog is not None and self.check_tracker is not None:
            self.check_tracker.stable_frames = camera_settings["stable_frames"]
            self.check_tracker.lift_area_ratio = camera_settings["lift_ratio"]
            self.check_tracker.update(0, track_input, self.hand_points, self.mp_hands_detector is not None)
        
        # Hitung rotasi / flip / pengangkatan balok (hanya saat tes berjalan)
        if self.game_started and 1 <= self.current_question <= LEVEL_COUNT:
            self.rotation_tracker.update(self.current_question, track_input, self.hand_points, self.mp_hands_detector is not None)
        if SHOW_ROTATION_COUNT:
            rot_n, flip_n, hid_n = self.rotation_tracker.level_counts(self.current_question)
            cv2.putText(frame, f"Rotasi: {rot_n}  Flip: {flip_n}  Angkat*: {hid_n}", (130, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    def _check_level(self, frame, box_design, box_distance, pos_x, pos_y):
        if len(box_design) == 4 and len(box_distance) == 4:
        
            # Draw lines connecting boxes (optimized with polylines)
            pts = np.array([
                [pos_x[0], pos_y[0]],
                [pos_x[1], pos_y[1]],
                [pos_x[2], pos_y[2]],
                [pos_x[3], pos_y[3]]
                ], np.int32)
            # Draw all connecting lines at once
            for i in range(4):
                for j in range(i+1, 4):
                    cv2.line(frame, tuple(pts[i]), tuple(pts[j]), (0, 0, 0), 2)
        
            #pos_x_order = [ pos_x[0], pos_x[1], pos_x[2], pos_x[3] ]
            #pos_y_order = [ pos_y[0], pos_y[1], pos_y[2], pos_y[3] ]
        
            len_0 = int (math.sqrt( (pos_x[0]-pos_x[1])**2 + (pos_y[0]-pos_y[1])**2 ) )
            len_1 = int (math.sqrt( (pos_x[1]-pos_x[2])**2 + (pos_y[1]-pos_y[2])**2 ) )
            len_2 = int (math.sqrt( (pos_x[2]-pos_x[3])**2 + (pos_y[2]-pos_y[3])**2 ) )
            len_3 = int (math.sqrt( (pos_x[3]-pos_x[0])**2 + (pos_y[3]-pos_y[0])**2 ) )
            len_4 = int (math.sqrt( (pos_x[0]-pos_x[2])**2 + (pos_y[0]-pos_y[2])**2 ) )
            len_5 = int (math.sqrt( (pos_x[1]-pos_x[3])**2 + (pos_y[1]-pos_y[3])**2 ) )
        
                    # Order Len
            len_order = [len_0, len_1, len_2, len_3, len_4, len_5]
            len_rect = sorted(len_order)
        
                    #print(len_rect)
        
            if (abs(len_rect[0] - len_rect[1]) < 100) and \
                (abs(len_rect[0] - len_rect[2]) < 100) and \
                (abs(len_rect[0] - len_rect[3]) < 100) and \
                (abs(len_rect[1] - len_rect[2]) < 100) and \
                (abs(len_rect[1] - len_rect[3]) < 100) and \
                (abs(len_rect[2] - len_rect[3]) < 100):
        
                    # 1. Gabungkan posisi x, y, dan index asli
                    indexed_positions = [(pos_x[i], pos_y[i], i) for i in range(len(pos_x))]
        
                    # 2. Urutkan berdasarkan x untuk memisahkan kelompok kiri dan kanan
                    sorted_by_x = sorted(pos_x)
                    mid_point = (sorted_by_x[1] + sorted_by_x[2]) / 2  # Titik tengah antara x terbesar kedua dan terkecil kedua
        
                    # 3. Urutkan: kelompok kiri (x < mid_point) dulu, lalu kelompok kanan, dalam setiap kelompok urutkan berdasarkan y
                    indexed_positions.sort(key=lambda p: (p[0] >= mid_point, p[1]))
        
                    # 4. Ambil index asli yang sudah terurut
                    sort_index = [idx for (x, y, idx) in indexed_positions]
        
                    # 5. Urutkan box_design berdasarkan index yang sudah diurutkan
                    box_design_sort = [box_design[i] for i in sort_index]
        
                    current_answer = LEVEL_ANSWERS.get(self.current_variant, []) if self.game_started else []
        
                    if (self.game_started and 1 <= self.current_question <= LEVEL_COUNT
                            and box_design_sort == current_answer):
                        self._on_level_solved(self.current_question)

    def _poll_yolo(self, frame):
        # Make detections using background thread (non-blocking)
        self.frame_count += 1
                
         # Only submit frame to YOLO every N frames to reduce load
        if self.frame_count % (self.yolo_skip_frames + 1) == 0:
            if self.yolo_thread.frame_queue.empty():  # Only if queue is empty
                try:
                    frame_for_yolo = np.array(frame, copy=True)
                    self.yolo_thread.frame_queue.put_nowait(frame_for_yolo)
                except:
                    pass  # Queue full, skip this frame
                
        # Get latest detection results if available (non-blocking)
        if not self.yolo_thread.result_queue.empty():
            try:
                self.latest_detections = self.yolo_thread.result_queue.get_nowait()
            except:
                pass
                
        # Use latest detections (may be from previous frame, but that's OK)
        return self.latest_detections

    def _process_hands(self, frame):
        # Use persistent MediaPipe Hands instance (not recreated every frame)
        # Skip MediaPipe on same frames as YOLO to reduce load
        if self.mp_hands_detector is not None and self.frame_count % (self.mediapipe_skip_frames + 1) == 0:
            # frame_rgb already in RGB format, no need to convert
            hand_result = self.mp_hands_detector.process(frame)
        
            if hand_result and hand_result.multi_hand_landmarks:
                fh, fw = frame.shape[:2]
                self.hand_points = [(lm.x * fw, lm.y * fh)
                                    for hl in hand_result.multi_hand_landmarks for lm in hl.landmark]
                for hand_landmarks in hand_result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_style, connection_style)
            else:
                self.hand_points = []

    def _render_preview(self, frame):
        if not HIDE_CAMERA and self.camera is not None and not self.game_started:
            # Convert frame to ImageTk format
            # img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = Image.fromarray(frame)
                    
            # Calculate aspect ratio
            frame_height, frame_width = frame.shape[:2]
            container_width = self.video_frame_1.winfo_width()
            container_height = self.video_frame_1.winfo_height()
            # Guard: jika widget belum siap (ukuran masih 1 atau 0), skip resize kali ini
            if container_width <= 1 or container_height <= 1:
                ImgTks = ImageTk.PhotoImage(image=img)
                self.camera.imgtk = ImgTks
                self.camera.configure(image=ImgTks)
                self.after(10, self.streaming)
                return True
                    
            # Calculate scaling factor while maintaining aspect ratio
            width_ratio = container_width / frame_width
            height_ratio = container_height / frame_height
            scale = min(width_ratio, height_ratio)
                    
            # Resize image if needed
            if scale < 1:
                new_width = int(frame_width * scale)
                new_height = int(frame_height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
            ImgTks = ImageTk.PhotoImage(image=img)
            self.camera.imgtk = ImgTks
            self.camera.configure(image=ImgTks)
        return False

    def _update_fps(self):
        # FPS monitoring
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
                    
            # Enhanced FPS output with debug info
            if self.debug_mode:
                queue_size = self.yolo_thread.frame_queue.qsize()
                result_size = self.yolo_thread.result_queue.qsize()
                last_inference = self.yolo_thread.last_inference_time
                print(f"FPS: {self.current_fps} | YOLO: {last_inference:.1f}ms | Queue: {queue_size}/{result_size}")
            else:
                print(f"FPS: {self.current_fps}")
                    
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def cleanup(self):
        """Cleanup resources before closing"""
        print(">>> Cleaning up resources...")
        if hasattr(self, 'yolo_thread'):
            self.yolo_thread.stop()
        if hasattr(self, 'serial_thread') and self.serial_thread:
            self.serial_thread.stop()
        if hasattr(self, 'mp_hands_detector') and self.mp_hands_detector is not None:
            self.mp_hands_detector.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def destroy(self):
        """Override destroy to cleanup resources"""
        self.cleanup()
        super().destroy()



if __name__ == "__main__":
    try:
        app = TimeIn()
        #app.streaming()
        app.after(0, lambda:app.state('zoomed'))
        app.mainloop()
    except Exception:
        print(">>> Application crashed. Full traceback:")
        traceback.print_exc()
        raise SystemExit(1)
