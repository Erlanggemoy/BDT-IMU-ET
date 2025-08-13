#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Import

import numpy as np
import cv2                                                                        
import os
import math
import time
import socket
                                                        
#
# \import pyautogui
#screen_width, screen_height = pyautogui.size()
#pyautogui.moveTo(screen_width, screen_height)
# Press the Windows key and the D key simultaneously to minimize all windows
#pyautogui.hotkey('win', 'd')


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Learning

import torch
 
# Device configuration
print("GPU CUDA is Available : " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_MODEL = os.path.join(BASE_DIR, 'MODEL', 'exp7', 'weights', 'best.pt')
model_yolo = torch.hub.load(os.path.join(BASE_DIR, 'MODEL', 'yolov5'), 'custom', path=PATH_MODEL, force_reload=True, source='local')
model_yolo.to(device)


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Read Files

task_00 = cv2.imread(os.path.join(BASE_DIR, 'FILES', 'TEST_1000x1000', '00.jpg'))
task_0A = cv2.imread(os.path.join(BASE_DIR, 'FILES', 'TEST_1000x1000', '0A.jpg'))
task_0B = cv2.imread(os.path.join(BASE_DIR, 'FILES', 'TEST_1000x1000', '0Bx.jpg'))
task_01 = os.path.join(BASE_DIR, 'FILES', 'TEST_1000x1000', '01.jpg')
task_02 = os.path.join(BASE_DIR, 'FILES', 'TEST_1000x1000', '02.jpg')
task_03 = os.path.join(BASE_DIR, 'FILES', 'TEST_1000x1000', '03.jpg')
task_04 = os.path.join(BASE_DIR, 'FILES', 'TEST_1000x1000', '04.jpg')
task_05 = os.path.join(BASE_DIR, 'FILES', 'TEST_1000x1000', '05.jpg')
task_06 = os.path.join(BASE_DIR, 'FILES', 'TEST_1000x1000', '06.jpg')
task_07 = os.path.join(BASE_DIR, 'FILES', 'TEST_1000x1000', '07.jpg')
task_08 = os.path.join(BASE_DIR, 'FILES', 'TEST_1000x1000', '08.jpg')
task_09 = cv2.imread(os.path.join(BASE_DIR, 'FILES', 'TEST_1000x1000', '09.jpg'))

#scale_percent = 70 # percent of original size
#task_width = int(task_0B.shape[1] * scale_percent / 100)
#task_height = int(task_0B.shape[0] * scale_percent / 100)
#task_dim = (task_width, task_height)

#task_resized = cv2.resize(task_0B, task_dim, interpolation = cv2.INTER_AREA)
#cv2.imshow('Task', task_0B)


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



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Connect to Database Server 
  
# PostgreSQL Setup
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
from pathlib import Path

def get_db_connection():
    """Create and return a PostgreSQL database connection using environment variables."""
    try:
        # Load environment variables from .env file
        env_path = Path('.') / '.env'
        load_dotenv(dotenv_path=env_path)
        
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
        
        # Set timezone to UTC+9 (Japan Standard Time)
        with conn.cursor() as cursor:
            cursor.execute("SET timezone = 'Asia/Tokyo';")
            
        print(">>> PostgreSQL database connected successfully.")
        print(">>> Timezone set to Asia/Tokyo (UTC+9)")
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


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Global Parameter

n_frame = 0    
n_capture = 3            # Detail 1 Normal 3  # Realtime 7
#n_contour = 0
n_test = 0

timer_return = 0

t_thumb_all = []

t_thumb_01 = 0
t_thumb_02 = 0
t_thumb_03 = 0
t_thumb_03 = 0
t_thumb_04 = 0
t_thumb_05 = 0
t_thumb_06 = 0
t_thumb_07 = 0
t_thumb_08 = 0

thumb_flag_01 = True
thumb_flag_02 = True
thumb_flag_03 = True
thumb_flag_04 = True
thumb_flag_05 = True
thumb_flag_06 = True
thumb_flag_07 = True
thumb_flag_08 = True

ans_01  = [2,1,1,2]
ans_02  = [1,3,1,1]
ans_02x = [1,1,3,1]
ans_03  = [2,2,3,4]
ans_03x = [2,3,2,4]
ans_04  = [5,1,4,1]
ans_04x = [5,4,1,1]
ans_05  = [4,3,5,6]
ans_05x = [4,5,3,6]
ans_06  = [1,4,6,1]
ans_06x = [1,6,4,1]
ans_07  = [5,6,4,3]
ans_07x = [5,4,6,3]
ans_08  = [5,3,4,5]
ans_08x = [5,4,3,5]

grasp_pose = [0,0,0,0,0,0,0]

speed_eye = 0
speed_hand = 0

scale_percent = 130 # percent of original size
task_width = int(task_0A.shape[1] * scale_percent / 100)
task_height = int(task_0A.shape[0] * scale_percent / 100)
task_dim = (task_width, task_height)
task_resized = cv2.resize(task_0A, task_dim, interpolation = cv2.INTER_AREA)

nick_name = ""
gender_code = ""
age_range_code = 0



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Text to  

import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')

#for voice in voices:
    #print(voice, voice.id)
    #engine.setProperty('voice', voice.id)

rate = engine.getProperty('rate')
engine.setProperty('rate', rate)

def SpeakText(command):
    newVoiceRate = 150
    engine.setProperty('rate',newVoiceRate)
    engine.say(command)
    engine.runAndWait()


from gtts import gTTS
from io import BytesIO
import pygame

def SpeakTextG(text):

    # Konversi teks ke suara
    tts = gTTS(text=text, lang="ja")
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)

    # Inisialisasi pygame untuk memutar audio
    pygame.mixer.init()
    audio_fp.seek(0)
    pygame.mixer.music.load(audio_fp)
    pygame.mixer.music.play()

    # Tunggu hingga audio selesai
    while pygame.mixer.music.get_busy():
        continue

#SpeakTextG("こんにちは、お元気ですか？")



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

        SpeakTextG("ブロックデザインゲームえようこそ。") #名前、年齢、性別を入力してください。
        #SpeakTextG("... ")


    def button_callback(self):
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
            import tkinter.messagebox as messagebox
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            print(f"Unexpected error: {e}")
            messagebox.showerror("Error", "Terjadi kesalahan saat memproses input")
        

app_input = App_Input()
app_input.mainloop()




#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Setup GUI Video

#PATH_VIDEO = r"C:\Users\50575\Desktop\BDT\VIDEO_EXPERIMENT\TLL\TOP_VIEW\anom_02.mp4"
#cap = cv2.VideoCapture(PATH_VIDEO)          #0, cv2.CAP_DSHOW

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)      #1280   640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)     #720    480

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



class TimeIn(customtkinter.CTk):

    def __init__(self):

        super().__init__()
        self.title("Block Design Test")

        self.geometry("960x560")
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(0, weight=1)


        # --------------------------------------------------- Design

        self.design_frame_0 = TextFrame(self, "Block Design")
        self.design_frame_0.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

        self.design_frame_1 = ImageFrame(self, "")
        self.design_frame_1.grid(row=1, column=0, padx=0, pady=0, sticky="nsew") # row=1

        IMAGE_WIDTH = 525 #650
        IMAGE_HEIGHT = 525 #650
        IMAGE_PATH = os.path.join(BASE_DIR, 'FILES', 'TEST_1000x1000', '0Bx.jpg')
        self.image = customtkinter.CTkImage(light_image=Image.open(IMAGE_PATH), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
        self.label = customtkinter.CTkLabel(master=self.design_frame_1, image=self.image, text='')
        self.label.grid(column=0, row=0)

        # --------------------------------------------------- Camera

        self.video_frame_0 = TextFrame(self, "Upper Table Camera")
        self.video_frame_0.grid(row=0, column=1, padx=0, pady=0, sticky="nsew")
        
        self.video_frame_1 = ImageFrame(self, "")
        self.video_frame_1.grid(row=1, column=1, padx=0, pady=0, sticky="nsew") #row=1
        
        self.camera = customtkinter.CTkLabel(self.video_frame_1, text="", anchor="center")
        self.camera.grid(row=0, column=0, padx=0, pady=0, sticky="nswe")


        # --------------------------------------------------- Camera

        self.button_0 = customtkinter.CTkButton(self, text="スタート (START)", font=("Helvetica", 30), command=self.button_0_callback)
        self.button_0.grid(row=2, column=0, columnspan=2, padx=65, pady=65, sticky="nswe") # , column=0, columnspan=2, row=2

        #self.button_1 = customtkinter.CTkButton(self, text="アプリ終了 (EXIT)", font=("Helvetica", 30), command=self.button_1_callback)
        #self.button_1.grid(row=2, column=1, padx=65, pady=65, sticky="nswe") # , columnspan=2, row=2

        #self.start_zero = time.time()
        #self.start_thumb = time.time()

        self.timer_task_all = []

        self.timer_task_01 = 0
        self.timer_task_02 = 0
        self.timer_task_03 = 0
        self.timer_task_04 = 0
        self.timer_task_05 = 0
        self.timer_task_06 = 0
        self.timer_task_07 = 0
        self.timer_task_08 = 0

        self.task_flag_01 = True
        self.task_flag_02 = True
        self.task_flag_03 = True
        self.task_flag_04 = True
        self.task_flag_05 = True
        self.task_flag_06 = True
        self.task_flag_07 = True
        self.task_flag_08 = True

        self.current_question = 1

        self.cognitive_age_list = []

        global nick_name
        global gender_code
        global age_range_code

        self.nick_name = nick_name
        self.gender_code = gender_code
        self.age_range_code = age_range_code

        self.retry_button = None

        #self.task_state = 1


        #SpeakTextG("赤と白の 4 つのブロックから表示されたデザインを作成してください。")
        #SpeakTextG("左手または利き手と反対の手で操作してください。")
        #SpeakTextG("準備ができたらスタートボタンを押してください。")

    

    def button_0_callback(self):

        # self.state = "disabled"
        self.button_0.grid_remove()
        self.show_current_level_button(self.current_question)
        
        SpeakTextG("こんにちわ")
        SpeakTextG("これから 絵合わせゲームで あなたの認知年齢を みてみましょう")
        SpeakTextG("3, 2, 1, スタート")
        print("START")
        self.start_task = time.time()
        

        IMAGE_WIDTH = 525 #650
        IMAGE_HEIGHT = 525 #650
        # IMAGE_PATH = os.path.join(BASE_DIR, 'FILES', 'TEST_1000x1000', '04.jpg')
        self.image = customtkinter.CTkImage(light_image=Image.open(task_01), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
        self.label = customtkinter.CTkLabel(master=self.design_frame_1, image=self.image, text='')
        self.label.grid(column=0, row=0)

        #self.start_task = time.time()
        self.streaming()


    #def button_1_callback(self):

        #os.execl(sys.executable, os.path.abspath(__file__), *sys.argv)
        #sys.exit() 



    def estimate_cognitive_age(self, time_finish_one_task):
        
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression

        # Transform the input features to higher-degree polynomials, Create a polynomial regression model
        # Fit the model to the transformed data, Predict y values for new x values

        degree = 2
        poly_features = PolynomialFeatures(degree=degree)

        #x_time      = np.array([13, 14, 15, 16, 17, 18, 22, 25, 30, 33, 37, 40, 45, 50]).reshape(-1, 1)
        
        x_time      = np.array([ 9, 10, 11, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50]).reshape(-1, 1)
        y_age       = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])

        time_avg = time_finish_one_task # * 1.5  # second

        x_time_poly = poly_features.fit_transform(x_time)
        model_time = LinearRegression()
        model_time.fit(x_time_poly, y_age)

        new_x_time = np.array([time_avg]).reshape(-1, 1)
        new_x_time_poly = poly_features.transform(new_x_time)
        cognitive_age = model_time.predict(new_x_time_poly)

        #print(cognitive_age)

        if (cognitive_age < 20):
            cognitive_age = 20
        elif (cognitive_age > 85):
            cognitive_age = 90
        
        cognitive_age = int(cognitive_age)

        return cognitive_age #.tolist()[0]

        #print("Estimated Cognitive Age : " + str(round(cognitive_age[0], 1)) + " years")

    def show_current_level_button(self, level):
        self.current_level_button = customtkinter.CTkButton(
            self,
            text="Level " + str(level),
            font=("Helvetica", 30)
        )
        
        self.current_level_button.grid(row=2, column=0, columnspan=2, padx=65, pady=65, sticky="nswe")
    
    def show_retry_button(self):
        self.retry_button = customtkinter.CTkButton(
            self,
            text="Retry Test",
            font=("Helvetica", 30),
            command=self.retry_test
        )
        
        # Position the retry button in the same place as the start button was
        self.retry_button.grid(row=2, column=0, columnspan=2, padx=65, pady=65, sticky="nswe")

    def retry_test(self):
        # Reset the application state
        self.destroy()  # Close the current window
        # Restart the application
        app_input = App_Input()
        app_input.mainloop()

        if nick_name:  # Only proceed if user provided input
            app = TimeIn()
            app.mainloop()

    # code for video streaming
    def streaming(self):

        self.button_0._state = "disabled"

        frame= cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (1000, 800))                 #(2000, 1600)   (1000, 800)

        # --------------------------------------------------------------------------- Start Process

        # Frame threshold 
        imgBlur = cv2.GaussianBlur(frame, (7,7), 1)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        ret, imgThres = cv2.threshold(imgGray, 175, 255, cv2.THRESH_BINARY) #175   195
        
        # Make detections 
        results = model_yolo(frame)

        df_tracked_objects = results.pandas().xyxy[0]
        list_tracked_objects = df_tracked_objects.values.tolist()

        #if len(list_tracked_objects) == 4: #>0

        IMAGE_WIDTH = 525   #650
        IMAGE_HEIGHT = 525  #650

        box_num = 0
        box_face = 0
        box_list = []
        box_design = []
        box_distance = []
        box_design_sort = []

        box_near_hand = []
        #avg_confidence = []

        pos_x = []
        pos_y = []

        for x1, y1, x2, y2, conf_pred, cls_id, cls in list_tracked_objects:

            if conf_pred > 0.7:

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
                
                dim = (100, 100)
                imgBox = cv2.resize(imgThres[y1:y2, x1:x2], dim, interpolation = cv2.INTER_AREA)
                #cv2.imshow("Box_"+str(box_num), imgBox)
                
                box_class = [ imgBox[50,25], imgBox[75,50], imgBox[50,75], imgBox[25,50] ]

                if box_class == [0,0,0,0] :
                    box_face = 1
                    box_design.append(1)
                elif box_class == [255,255,255,255]:
                    box_face = 2
                    box_design.append(2)
                #... dipisah
                elif box_class == [255,255,0,0]:
                    box_face = 3
                    box_design.append(3)  
                elif box_class == [255,0,0,255]:
                    box_face = 4
                    box_design.append(4)  
                elif box_class == [0,0,255,255]:
                    box_face = 5
                    box_design.append(5)  
                elif box_class == [0,255,255,0]:
                    box_face = 6
                    box_design.append(6)               
                
                cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), (0, 255, 0), 2)

                box_num = box_num + 1
                roi_label = frame[y1:y1+50, x1:x1+50]

                if(box_face == 1):
                    try:                            
                        roi_label [np.where(mask_face_01)] = 0
                        roi_label += face_01
                    except IndexError:
                        pass
                elif(box_face == 2):
                    try:                            
                        roi_label [np.where(mask_face_02)] = 0
                        roi_label += face_02
                    except IndexError:
                        pass
                elif(box_face == 3):
                    try:                            
                        roi_label [np.where(mask_face_03)] = 0
                        roi_label += face_03
                    except IndexError:
                        pass
                elif(box_face == 4):
                    try:                            
                        roi_label [np.where(mask_face_04)] = 0
                        roi_label += face_04
                    except IndexError:
                        pass
                elif(box_face == 5):
                    try:                            
                        roi_label [np.where(mask_face_05)] = 0
                        roi_label += face_05
                    except IndexError:
                        pass
                elif(box_face == 6):
                    try:                            
                        roi_label [np.where(mask_face_06)] = 0
                        roi_label += face_06
                    except IndexError:
                        pass

        
        # >>>>>>>>>>>>

        if len(box_design) == 4 and len(box_distance) == 4:

            box_0 = (pos_x[0], pos_y[0])
            box_1 = (pos_x[1], pos_y[1])
            box_2 = (pos_x[2], pos_y[2])
            box_3 = (pos_x[3], pos_y[3])

            frame = cv2.line(frame, box_0, box_1, (0, 0, 0), 2)
            frame = cv2.line(frame, box_0, box_2, (0, 0, 0), 2)
            frame = cv2.line(frame, box_0, box_3, (0, 0, 0), 2)
            frame = cv2.line(frame, box_1, box_2, (0, 0, 0), 2)
            frame = cv2.line(frame, box_1, box_3, (0, 0, 0), 2)
            frame = cv2.line(frame, box_2, box_3, (0, 0, 0), 2)

            #pos_x_order = [ pos_x[0], pos_x[1], pos_x[2], pos_x[3] ]
            #pos_y_order = [ pos_y[0], pos_y[1], pos_y[2], pos_y[3] ]

            len_0 = int (math.sqrt( (pos_x[0]-pos_x[1])**2 + (pos_y[0]-pos_y[1])**2 ) )
            len_1 = int (math.sqrt( (pos_x[1]-pos_x[2])**2 + (pos_y[1]-pos_y[2])**2 ) )
            len_2 = int (math.sqrt( (pos_x[2]-pos_x[3])**2 + (pos_y[2]-pos_y[3])**2 ) )
            len_3 = int (math.sqrt( (pos_x[3]-pos_x[0])**2 + (pos_y[3]-pos_y[0])**2 ) )
            len_4 = int (math.sqrt( (pos_x[0]-pos_x[2])**2 + (pos_y[0]-pos_y[2])**2 ) )
            len_5 = int (math.sqrt( (pos_x[1]-pos_x[3])**2 + (pos_y[1]-pos_y[3])**2 ) )

            # Order Len
            len_order = [ len_0, len_1, len_2, len_3, len_4, len_5 ]
            len_rect = sorted(len_order)

            #print(len_rect)

            if  ( abs(len_rect[0] - len_rect[1]) < 100) and \
                ( abs(len_rect[0] - len_rect[2]) < 100) and \
                ( abs(len_rect[0] - len_rect[3]) < 100) and \
                ( abs(len_rect[1] - len_rect[2]) < 100) and \
                ( abs(len_rect[1] - len_rect[3]) < 100) and \
                ( abs(len_rect[2] - len_rect[3]) < 100):

                # >>>>>>>>>>>>
                
                sort_index = sorted(range(len(box_distance)), key=lambda k: box_distance[k])

                for i in range(len(sort_index)):
                    box_design_sort.append(box_design[sort_index[i]])

                    
                if self.current_question == 1 and box_design_sort == ans_01:
                    test_label = frame[20:120, 20:120]
                    test_label [np.where(mask_img_01)] = 0
                    test_label += img_01
                    
                    #self.task_state = self.task_state + 1
                    
                    if(self.task_flag_01):
                        end_task = time.time()
                        self.timer_task_01 = round((end_task - self.start_task - timer_return), 2)
                        self.timer_task_all.append(self.timer_task_01)

                        self.cognitive_age_list.append(self.estimate_cognitive_age(self.timer_task_01))

                        print ("TASK 1 COMPLETED in " + str(self.timer_task_01) +" seconds")
                        SpeakTextG("問題 1 は" + str(int(self.timer_task_01)) +" 秒で できました。")

                        if int(self.timer_task_01) < 10:
                            SpeakTextG("さいこうだよ.")
                        elif int(self.timer_task_01) < 15:
                            SpeakTextG("すばらしい.")
                        elif int(self.timer_task_01) < 20:
                            SpeakTextG("グッジョブ！.")
                        elif int(self.timer_task_01) < 25:
                            SpeakTextG("よくできたね.")
                        elif int(self.timer_task_01) < 30:
                            SpeakTextG("ナイス.")
                        else:
                            SpeakTextG("がんばったね.")

                        SpeakTextG("次は問題 2 です。")

                        #IMAGE_WIDTH = 650
                        #IMAGE_HEIGHT = 650
                        self.image = customtkinter.CTkImage(light_image=Image.open(task_02), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
                        self.label = customtkinter.CTkLabel(master=self.design_frame_1, image=self.image, text='')
                        self.label.grid(column=0, row=0)

                        self.current_question = 2
                        self.current_level_button.grid_remove()
                        self.show_current_level_button(self.current_question)
                        self.task_flag_01 = False
                        self.start_task = time.time()

                    # Detect Thumb
                    """
                    if (thumb_state == 1 and thumb_flag_01 == True):
                        end_thumb = time.time()
                        timer_thumb = end_thumb - start_thumb
                        t_thumb_04 = round((timer_thumb - self.timer_task_04), 2)
                        t_thumb_all.append(t_thumb_04)

                        start_thumb = time.time()
                        thumb_flag_04 = False

                        timer_speak = self.timer_task_04
                        if timer_speak < 10:
                            SpeakTextG("Amazing.")
                        if timer_speak < 15:
                            SpeakTextG("Awesome.")
                        elif timer_speak < 20:
                            SpeakTextG("Great.")
                        elif timer_speak < 25:
                            SpeakTextG("Good job.")
                        elif timer_speak < 30:
                            SpeakTextG("Well done.")
                        else:
                            SpeakTextG("Give your best.")
                    """

                elif self.current_question == 2 and box_design_sort == ans_02:
                    test_label = frame[20:120, 20:120]
                    test_label [np.where(mask_img_02)] = 0
                    test_label += img_02
                    
                    #self.task_state = self.task_state + 1
                    
                    if(self.task_flag_02):
                        end_task = time.time()
                        self.timer_task_02 = round((end_task - self.start_task - timer_return), 2)
                        self.timer_task_all.append(self.timer_task_02)

                        self.cognitive_age_list.append(self.estimate_cognitive_age(self.timer_task_02))

                        print ("TASK 2 COMPLETED in " + str(self.timer_task_02) +" seconds")
                        SpeakTextG("問題 2 は" + str(int(self.timer_task_02)) +" 秒で できました。")

                        if int(self.timer_task_02) < 10:
                            SpeakTextG("さいこうだよ.")
                        elif int(self.timer_task_02) < 15:
                            SpeakTextG("すばらしい.")
                        elif int(self.timer_task_02) < 20:
                            SpeakTextG("グッジョブ！.")
                        elif int(self.timer_task_02) < 25:
                            SpeakTextG("よくできたね.")
                        elif int(self.timer_task_02) < 30:
                            SpeakTextG("ナイス.")
                        else:
                            SpeakTextG("がんばったね.")

                        SpeakTextG("次は問題 3 です。")

                        #IMAGE_WIDTH = 650
                        #IMAGE_HEIGHT = 650
                        self.image = customtkinter.CTkImage(light_image=Image.open(task_03), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
                        self.label = customtkinter.CTkLabel(master=self.design_frame_1, image=self.image, text='')
                        self.label.grid(column=0, row=0)

                        self.current_question = 3
                        self.current_level_button.grid_remove()
                        self.show_current_level_button(self.current_question)
                        self.task_flag_02 = False
                        self.start_task = time.time()

                    # Detect Thumb
                    """
                    if (thumb_state == 1 and thumb_flag_04 == True):
                        end_thumb = time.time()
                        timer_thumb = end_thumb - start_thumb
                        t_thumb_04 = round((timer_thumb - self.timer_task_04), 2)
                        t_thumb_all.append(t_thumb_04)

                        start_thumb = time.time()
                        thumb_flag_04 = False

                        timer_speak = self.timer_task_04
                        if timer_speak < 10:
                            SpeakTextG("Amazing.")
                        if timer_speak < 15:
                            SpeakTextG("Awesome.")
                        elif timer_speak < 20:
                            SpeakTextG("Great.")
                        elif timer_speak < 25:
                            SpeakTextG("Good job.")
                        elif timer_speak < 30:
                            SpeakTextG("Well done.")
                        else:
                            SpeakTextG("Give your best.")
                    """
                
                elif self.current_question == 3 and box_design_sort == ans_03:
                    test_label = frame[20:120, 20:120]
                    test_label [np.where(mask_img_03)] = 0
                    test_label += img_03
                    
                    #self.task_state = self.task_state + 1
                    
                    if(self.task_flag_03):
                        end_task = time.time()
                        self.timer_task_03 = round((end_task - self.start_task - timer_return), 2)
                        self.timer_task_all.append(self.timer_task_03)

                        self.cognitive_age_list.append(self.estimate_cognitive_age(self.timer_task_03))

                        print ("TASK 3 COMPLETED in " + str(self.timer_task_03) +" seconds")
                        SpeakTextG("問題 3 は" + str(int(self.timer_task_03)) +" 秒で できました。")

                        if int(self.timer_task_03) < 10:
                            SpeakTextG("さいこうだよ.")
                        elif int(self.timer_task_03) < 15:
                            SpeakTextG("すばらしい.")
                        elif int(self.timer_task_03) < 20:
                            SpeakTextG("グッジョブ！.")
                        elif int(self.timer_task_03) < 25:
                            SpeakTextG("よくできたね.")
                        elif int(self.timer_task_03) < 30:
                            SpeakTextG("ナイス.")
                        else:
                            SpeakTextG("がんばったね.")

                        SpeakTextG("次は問題 4 です。")

                        #IMAGE_WIDTH = 650
                        #IMAGE_HEIGHT = 650
                        self.image = customtkinter.CTkImage(light_image=Image.open(task_04), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
                        self.label = customtkinter.CTkLabel(master=self.design_frame_1, image=self.image, text='')
                        self.label.grid(column=0, row=0)

                        self.current_question = 4
                        self.current_level_button.grid_remove()
                        self.show_current_level_button(self.current_question)
                        self.task_flag_03 = False
                        self.start_task = time.time()

                    # Detect Thumb
                    """
                    if (thumb_state == 1 and thumb_flag_04 == True):
                        end_thumb = time.time()
                        timer_thumb = end_thumb - start_thumb
                        t_thumb_04 = round((timer_thumb - self.timer_task_04), 2)
                        t_thumb_all.append(t_thumb_04)

                        start_thumb = time.time()
                        thumb_flag_04 = False

                        timer_speak = self.timer_task_04
                        if timer_speak < 10:
                            SpeakTextG("Amazing.")
                        if timer_speak < 15:
                            SpeakTextG("Awesome.")
                        elif timer_speak < 20:
                            SpeakTextG("Great.")
                        elif timer_speak < 25:
                            SpeakTextG("Good job.")
                        elif timer_speak < 30:
                            SpeakTextG("Well done.")
                        else:
                            SpeakTextG("Give your best.")
                    """    

                elif self.current_question == 4 and box_design_sort == ans_04:
                    test_label = frame[20:120, 20:120]
                    test_label [np.where(mask_img_04)] = 0
                    test_label += img_04
                    
                    #self.task_state = self.task_state + 1
                    
                    if(self.task_flag_04):
                        end_task = time.time()
                        self.timer_task_04 = round((end_task - self.start_task - timer_return), 2)
                        self.timer_task_all.append(self.timer_task_04)

                        self.cognitive_age_list.append(self.estimate_cognitive_age(self.timer_task_04))

                        print ("TASK 4 COMPLETED in " + str(self.timer_task_04) +" seconds")
                        SpeakTextG("問題 4 は" + str(int(self.timer_task_04)) +" 秒で できました。")

                        if int(self.timer_task_04) < 10:
                            SpeakTextG("さいこうだよ.")
                        elif int(self.timer_task_04) < 15:
                            SpeakTextG("すばらしい.")
                        elif int(self.timer_task_04) < 20:
                            SpeakTextG("グッジョブ！.")
                        elif int(self.timer_task_04) < 25:
                            SpeakTextG("よくできたね.")
                        elif int(self.timer_task_04) < 30:
                            SpeakTextG("ナイス.")
                        else:
                            SpeakTextG("がんばったね.")

                        SpeakTextG("次は問題 5 です。")

                        #IMAGE_WIDTH = 650
                        #IMAGE_HEIGHT = 650
                        self.image = customtkinter.CTkImage(light_image=Image.open(task_05), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
                        self.label = customtkinter.CTkLabel(master=self.design_frame_1, image=self.image, text='')
                        self.label.grid(column=0, row=0)

                        self.current_question = 5
                        self.current_level_button.grid_remove()
                        self.show_current_level_button(self.current_question)
                        self.task_flag_04 = False
                        self.start_task = time.time()

                    # Detect Thumb
                    """
                    if (thumb_state == 1 and thumb_flag_04 == True):
                        end_thumb = time.time()
                        timer_thumb = end_thumb - start_thumb
                        t_thumb_04 = round((timer_thumb - self.timer_task_04), 2)
                        t_thumb_all.append(t_thumb_04)

                        start_thumb = time.time()
                        thumb_flag_04 = False

                        timer_speak = self.timer_task_04
                        if timer_speak < 10:
                            SpeakTextG("Amazing.")
                        if timer_speak < 15:
                            SpeakTextG("Awesome.")
                        elif timer_speak < 20:
                            SpeakTextG("Great.")
                        elif timer_speak < 25:
                            SpeakTextG("Good job.")
                        elif timer_speak < 30:
                            SpeakTextG("Well done.")
                        else:
                            SpeakTextG("Give your best.")
                    """


                elif self.current_question == 5 and box_design_sort == ans_05:
                    test_label = frame[20:120, 20:120]
                    test_label [np.where(mask_img_05)] = 0
                    test_label += img_05

                    #self.task_state = self.task_state + 1
                    
                    if(self.task_flag_05):
                        end_task = time.time()
                        self.timer_task_05 = round((end_task - self.start_task - timer_return), 2)
                        self.timer_task_all.append(self.timer_task_05)

                        self.cognitive_age_list.append(self.estimate_cognitive_age(self.timer_task_05))

                        print ("TASK 5 COMPLETED in " + str(self.timer_task_05) +" seconds")
                        SpeakTextG("問題 5 は  " + str(int(self.timer_task_05)) +" 秒できました。")

                        if int(self.timer_task_05) < 10:
                            SpeakTextG("さいこうだよ.")
                        elif int(self.timer_task_05) < 15:
                            SpeakTextG("すばらしい.")
                        elif int(self.timer_task_05) < 20:
                            SpeakTextG("グッジョブ！.")
                        elif int(self.timer_task_05) < 25:
                            SpeakTextG("よくできたね.")
                        elif int(self.timer_task_05) < 30:
                            SpeakTextG("ナイス.")
                        else:
                            SpeakTextG("がんばったね.")

                        SpeakTextG("次は問題 6 です。")

                        #IMAGE_WIDTH = 650
                        #IMAGE_HEIGHT = 650
                        self.image = customtkinter.CTkImage(light_image=Image.open(task_06), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
                        self.label = customtkinter.CTkLabel(master=self.design_frame_1, image=self.image, text='')
                        self.label.grid(column=0, row=0)

                        self.current_question = 6
                        self.current_level_button.grid_remove()
                        self.show_current_level_button(self.current_question)
                        self.task_flag_05 = False
                        self.start_task = time.time()
                    
                    # Detect Thumb
                    """
                    if (thumb_state == 1 and thumb_flag_05 == True):
                        end_thumb = time.time()
                        timer_thumb = end_thumb - start_thumb
                        t_thumb_05 = round((timer_thumb - self.timer_task_05), 2)
                        t_thumb_all.append(t_thumb_05)

                        start_thumb = time.time()
                        thumb_flag_05 = False

                        timer_speak = self.timer_task_05
                        if timer_speak < 10:
                            SpeakTextG("Amazing.")
                        if timer_speak < 15:
                            SpeakTextG("Awesome.")
                        elif timer_speak < 20:
                            SpeakTextG("Great.")
                        elif timer_speak < 25:
                            SpeakTextG("Good job.")
                        elif timer_speak < 30:
                            SpeakTextG("Well done.")
                        else:
                            SpeakTextG("Give your best.")
                    """


                elif self.current_question == 6 and box_design_sort == ans_06:
                    test_label = frame[20:120, 20:120]
                    test_label [np.where(mask_img_06)] = 0
                    test_label += img_06

                    #self.task_state = self.task_state + 1
                    
                    if(self.task_flag_06):
                        end_task = time.time()
                        self.timer_task_06 = round((end_task - self.start_task - timer_return), 2)
                        self.timer_task_all.append(self.timer_task_06)

                        self.cognitive_age_list.append(self.estimate_cognitive_age(self.timer_task_06))

                        print ("TASK 6 COMPLETED in " + str(self.timer_task_06) +" seconds")
                        SpeakTextG("問題 6 は  " + str(int(self.timer_task_06)) +" 秒できました。")

                        if int(self.timer_task_06) < 10:
                            SpeakTextG("さいこうだよ.")
                        elif int(self.timer_task_06) < 15:
                            SpeakTextG("すばらしい.")
                        elif int(self.timer_task_06) < 20:
                            SpeakTextG("グッジョブ！.")
                        elif int(self.timer_task_06) < 25:
                            SpeakTextG("よくできたね.")
                        elif int(self.timer_task_06) < 30:
                            SpeakTextG("ナイス.")
                        else:
                            SpeakTextG("がんばったね.")

                        SpeakTextG("次は問題 7 です。")

                        #IMAGE_WIDTH = 650
                        #IMAGE_HEIGHT = 650
                        self.image = customtkinter.CTkImage(light_image=Image.open(task_07), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
                        self.label = customtkinter.CTkLabel(master=self.design_frame_1, image=self.image, text='')
                        self.label.grid(column=0, row=0)

                        self.current_question = 7
                        self.current_level_button.grid_remove()
                        self.show_current_level_button(self.current_question)
                        self.task_flag_06 = False
                        self.start_task = time.time()

                    # Detect Thumb
                    """
                    if (thumb_state == 1 and thumb_flag_06 == True):
                        end_thumb = time.time()
                        timer_thumb = end_thumb - start_thumb
                        t_thumb_06 = round((timer_thumb - self.timer_task_06), 2)
                        t_thumb_all.append(t_thumb_06)

                        start_thumb = time.time()
                        thumb_flag_06 = False
                    
                        timer_speak = self.timer_task_06
                        if timer_speak < 10:
                            SpeakTextG("Amazing.")
                        if timer_speak < 15:
                            SpeakTextG("Awesome.")
                        elif timer_speak < 20:
                            SpeakTextG("Great.")
                        elif timer_speak < 25:
                            SpeakTextG("Good job.")
                        elif timer_speak < 30:
                            SpeakTextG("Well done.")
                        else:
                            SpeakTextG("Give your best.")
                    """

                elif self.current_question == 7 and box_design_sort == ans_07:
                    test_label = frame[20:120, 20:120]
                    test_label [np.where(mask_img_07)] = 0
                    test_label += img_07
                    
                    #self.task_state = self.task_state + 1
                    
                    if(self.task_flag_07):
                        end_task = time.time()
                        self.timer_task_07 = round((end_task - self.start_task - timer_return), 2)
                        self.timer_task_all.append(self.timer_task_07)

                        self.cognitive_age_list.append(self.estimate_cognitive_age(self.timer_task_07))

                        print ("TASK 7 COMPLETED in " + str(self.timer_task_07) +" seconds")
                        SpeakTextG("問題 7 は" + str(int(self.timer_task_07)) +" 秒で できました。")

                        if int(self.timer_task_07) < 10:
                            SpeakTextG("さいこうだよ.")
                        elif int(self.timer_task_07) < 15:
                            SpeakTextG("すばらしい.")
                        elif int(self.timer_task_07) < 20:
                            SpeakTextG("グッジョブ！.")
                        elif int(self.timer_task_07) < 25:
                            SpeakTextG("よくできたね.")
                        elif int(self.timer_task_07) < 30:
                            SpeakTextG("ナイス.")
                        else:
                            SpeakTextG("がんばったね.")

                        SpeakTextG("次は問題 8 です。")

                        #IMAGE_WIDTH = 650
                        #IMAGE_HEIGHT = 650
                        self.image = customtkinter.CTkImage(light_image=Image.open(task_08), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
                        self.label = customtkinter.CTkLabel(master=self.design_frame_1, image=self.image, text='')
                        self.label.grid(column=0, row=0)

                        self.current_question = 8
                        self.current_level_button.grid_remove()
                        self.show_current_level_button(self.current_question)
                        self.task_flag_07 = False
                        self.start_task = time.time()

                    # Detect Thumb
                    """
                    if (thumb_state == 1 and thumb_flag_04 == True):
                        end_thumb = time.time()
                        timer_thumb = end_thumb - start_thumb
                        t_thumb_04 = round((timer_thumb - self.timer_task_04), 2)
                        t_thumb_all.append(t_thumb_04)

                        start_thumb = time.time()
                        thumb_flag_04 = False

                        timer_speak = self.timer_task_04
                        if timer_speak < 10:
                            SpeakTextG("Amazing.")
                        if timer_speak < 15:
                            SpeakTextG("Awesome.")
                        elif timer_speak < 20:
                            SpeakTextG("Great.")
                        elif timer_speak < 25:
                            SpeakTextG("Good job.")
                        elif timer_speak < 30:
                            SpeakTextG("Well done.")
                        else:
                            SpeakTextG("Give your best.")
                    """

                elif self.current_question == 8 and box_design_sort == ans_08:
                    test_label = frame[20:120, 20:120]
                    test_label [np.where(mask_img_08)] = 0
                    test_label += img_08

                    #self.task_state = self.task_state + 1
                    
                    if(self.task_flag_08):
                        end_task = time.time()
                        self.timer_task_08 = round((end_task - self.start_task - timer_return), 2)
                        self.timer_task_all.append(self.timer_task_08)

                        self.cognitive_age_list.append(self.estimate_cognitive_age(self.timer_task_08))

                        print ("TASK 8 COMPLETED in " + str(self.timer_task_08) +" seconds")
                        SpeakTextG("問題 8 は  " + str(int(self.timer_task_08)) +" 秒できました。")

                        if int(self.timer_task_08) < 10:
                            SpeakTextG("さいこうだよ.")
                        elif int(self.timer_task_08) < 15:
                            SpeakTextG("すばらしい.")
                        elif int(self.timer_task_08) < 20:
                            SpeakTextG("グッジョブ！.")
                        elif int(self.timer_task_08) < 25:
                            SpeakTextG("よくできたね.")
                        elif int(self.timer_task_08) < 30:
                            SpeakTextG("ナイス.")
                        else:
                            SpeakTextG("がんばったね.")

                        #IMAGE_WIDTH = 650
                        #IMAGE_HEIGHT = 650
                        # IMAGE_PATH = os.path.join(BASE_DIR, 'FILES', 'TEST_1000x1000', '08.jpg')
                        # self.image = customtkinter.CTkImage(light_image=Image.open(IMAGE_PATH), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
                        # self.label = customtkinter.CTkLabel(master=self.design_frame_1, image=self.image, text='')
                        # self.label.grid(column=0, row=0)

                        self.task_flag_08 = False
                        self.current_level_button.grid_remove()
                        # self.start_task = time.time()
                    
                        self.timer_task_avg = format(float(sum(self.timer_task_all) / len(self.timer_task_all)), ".3f")
                        cognitive_age_avg = int(sum(self.cognitive_age_list) / len(self.cognitive_age_list))
                        #t_thumb_avg = int(sum(t_thumb_all)) / len(t_thumb_all)

                        print("-------------------------------------------")
                        print("BLOCK DESIGN TEST ")
                        print("-------------------------------------------")
                        print("GRASP_POSTURE : " + str(grasp_pose))
                        print("TIMER_TASK    : " + str(self.timer_task_all))
                        print("TIMER_THUMB   : " + str(t_thumb_all))
                        print("COGNITIVE_AGE : " + str(self.cognitive_age_list))
                        print("-------------------------------------------")

                        age_real = int(self.age_range_code)
                        age_cog = cognitive_age_avg

                        visuo_spatial = 100

                        if (age_cog <= age_real):
                            visuo_spatial = 100 
                        else:
                            visuo_spatial = 100 - (age_cog-age_real)

                        print("Your name is " + self.nick_name)
                        print("Your gender is " + self.gender_code)
                        print("Your Average Completed Time is " + str( self.timer_task_avg) +" second")
                        print("Your Current Age is " + str( age_real ) +" years")
                        print("Your Cognitive Age is " + str( age_cog ) +" years")
                        print("Your Cognitive Fitness is " + str( visuo_spatial ) +" %")

                        SpeakTextG("これで、ブロックデザインゲームは終了です。")
                        #SpeakTextG("平均完了時間は " + str( self.timer_task_avg) +" 秒です")
                        #SpeakTextG("あなたの現在の年齢は " + str( age_real) +" 歳です")
                        SpeakTextG("あなたの推定認知年齢は " + str( age_cog ) +" 歳です")
                        #SpeakTextG("あなたの認知能力は " + str(visuo_spatial ) +" %")
                        
                       # SpeakTextG("画面上の鍵でフクロウを入手できます。ありがとうございます。") #お疲れさまでした。

                        time.sleep(3)

                        # Save results to PostgreSQL database
                        try:
                            conn = get_db_connection()
                            if conn:
                                with conn.cursor() as cursor:
                                    # Prepare the SQL query with parameterized values
                                    insert_query = """
                                    INSERT INTO bdt.participant_data (
                                        name, 
                                        real_age, 
                                        gender, 
                                        estimated_cognitive_age, 
                                        cognitive_fitness, 
                                        avg_time_all,
                                        avg_time_1,
                                        avg_time_2,
                                        avg_time_3,
                                        avg_time_4,
                                        avg_time_5,
                                        avg_time_6,
                                        avg_time_7,
                                        avg_time_8
                                    ) VALUES (
                                        %s, %s, %s, %s, %s, %s,
                                        %s, %s, %s, %s, %s, %s, %s, %s
                                    )
                                    """
                                    
                                    # Prepare the values tuple
                                    values = (
                                        self.nick_name,                      # name
                                        int(self.age_range_code),           # real_age
                                        self.gender_code,                   # gender
                                        int(age_cog),                       # estimated_cognitive_age
                                        f"{visuo_spatial}%",                # cognitive_fitness
                                        float(self.timer_task_avg),         # avg_time_all
                                        float(self.timer_task_all[0]) if len(self.timer_task_all) > 0 else None,  # avg_time_1
                                        float(self.timer_task_all[1]) if len(self.timer_task_all) > 1 else None,  # avg_time_2
                                        float(self.timer_task_all[2]) if len(self.timer_task_all) > 2 else None,  # avg_time_3
                                        float(self.timer_task_all[3]) if len(self.timer_task_all) > 3 else None,  # avg_time_4
                                        float(self.timer_task_all[4]) if len(self.timer_task_all) > 4 else None,  # avg_time_5
                                        float(self.timer_task_all[5]) if len(self.timer_task_all) > 5 else None,  # avg_time_6
                                        float(self.timer_task_all[6]) if len(self.timer_task_all) > 6 else None,  # avg_time_7
                                        float(self.timer_task_all[7]) if len(self.timer_task_all) > 7 else None   # avg_time_8
                                    )
                                    
                                    # Execute the query
                                    cursor.execute(insert_query, values)
                                    conn.commit()
                                    print(">>> Successfully saved participant data to database")
                                
                                # Close the connection
                                conn.close()
                            
                        except Exception as e:
                            print(">>> Error saving to database:")
                            print(f"Error: {e}")
                            if 'conn' in locals():
                                conn.rollback()
                        finally:
                            if 'conn' in locals() and conn is not None:
                                conn.close()

                        #if cognitive_age_avg <= 30:
                        #    SpeakTextG("あなたの推定認知年齢は 30 歳未満です")
                        #elif cognitive_age_avg > 30 and cognitive_age_avg <= 40 :
                        #    SpeakTextG("あなたの推定認知年齢は 40 歳から 50 歳の間です")
                        #elif cognitive_age_avg > 40 and cognitive_age_avg <= 50 :
                        #    SpeakTextG("あなたの推定認知年齢は 40 歳から 50 歳の間です")
                        #elif cognitive_age_avg > 50 and cognitive_age_avg <= 60 :
                        #    SpeakTextG("あなたの推定認知年齢は 50 歳から 60 歳の間です")
                        #elif cognitive_age_avg > 60 and cognitive_age_avg <= 70 :
                        #    SpeakTextG("あなたの推定認知年齢は 60 歳から 70 歳の間です")
                        #elif cognitive_age_avg > 70 :
                        #    SpeakTextG("あなたの推定認知年齢は 70 歳を超えています")

                        #SpeakTextG("ブロックゲームにご参加いただきありがとうございます。")
                        #SpeakTextG("また会いましょう、さようなら。")
                        
                        #>>>>>>>>>>>>>>>>>>>>>>

                        #IMAGE_WIDTH = 650
                        #IMAGE_HEIGHT = 650
                        IMAGE_PATH = os.path.join(BASE_DIR, 'FILES', 'TEST_1000x1000', '09.jpg')
                        self.image = customtkinter.CTkImage(light_image=Image.open(IMAGE_PATH), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
                        self.label = customtkinter.CTkLabel(master=self.design_frame_1, image=self.image, text='')
                        self.label.grid(column=0, row=0)

                        self.task_flag_08 = False
                        
                        self.show_retry_button()

                        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        #break

                    # Detect Thumb
                    """
                    if (thumb_state == 1 and thumb_flag_08 == True):
                        end_thumb = time.time()
                        timer_thumb = end_thumb - start_thumb
                        t_thumb_08 = round((timer_thumb - self.timer_task_08), 2)
                        t_thumb_all.append(t_thumb_08)

                        start_thumb = time.time()
                        thumb_flag_08 = False

                        timer_speak = self.timer_task_08
                        if timer_speak < 10:
                            SpeakTextG("Congratulations you did it !")
                    """
                else:
                    #print ("NOT COMPLETE")
                    pass

                box_design = []
                box_distance = []
                box_design_sort = []
    


        # --------------------------------------------------------------------------- End Process

        img = Image.fromarray(frame)
        ImgTks = ImageTk.PhotoImage(image=img)
        self.camera.imgtk = ImgTks
        self.camera.configure(image=ImgTks)
        self.camera.grid(column=0, row=0, padx=0, pady=0, sticky="nswe")

        self.after(20, self.streaming)



if __name__ == "__main__":
    app = TimeIn()
    #app.streaming()
    app.after(0, lambda:app.state('zoomed'))
    app.mainloop()
