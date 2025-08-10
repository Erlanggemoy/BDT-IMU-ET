#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Import

import numpy as np
import cv2
import os
import sys
import math
import time
import socket

import threading
import time

class ServerConnection:
    serverIPAddress = None
    SERVER_TCP_PORT = 9866
    BUFFER_SIZE = 1024
    
    connectionThread = None
    commandThread = None

    
    def __enter__(self):
        return self
 
    def __exit__(self, *args):
        self.release()
        
    # Receive data from server
    def receivedData(self, data):
        print(data)
        pass
    
    def getUDPServer(self):        
        if self.connectionThread is None:
            self.connectionThread = threading.Thread(target = self._getServerIPAddress)
            self.connectionThread.start()      
    
    def _getServerIPAddress(self):
        print("Trying to obtains server address.")
        UDP_IP = ""
        UDP_PORT = 9000
        ipAddress = None
        countError = 0
        while ipAddress is None and countError < 10:
            try:
                sock = socket.socket(socket.AF_INET, # Internet
                                     socket.SOCK_DGRAM) # UDP
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((UDP_IP, UDP_PORT))
               
                data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
                ipAddress = data.decode("utf-8")
                print("Server IP: {}".format(data))
            except Exception as e:
                print('Request Server IP Error')
                print(e)
                countError += 1
        sock = None
        if ipAddress is not None:
            self.serverIPAddress = ipAddress            
        self.connectionThread = None
        
    def sendTCPIP(self, data) :
        
        if self.connectionThread is None:
            self.connectionThread = threading.Thread(target=self._sendTCPIP, args=(data,))
            self.connectionThread.start()
      
    def _sendTCPIP(self, data, returnSuccess = None):
        sendResult = False
        try:
            server_sock =  socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.connect((self.serverIPAddress, self.SERVER_TCP_PORT))
                    
            MESSAGE = str.encode(data)
            print("Start Sending to Server")
            server_sock.send(MESSAGE)
            print("Start Receive Data From Server")      
            data = server_sock.recv(self.BUFFER_SIZE)
            print("received data from Server:", data)        
            data = data.decode("utf-8")    
            self.receivedData(data)    
            server_sock.close()
            sendResult = True
        except Exception as e:
            print("Send TCP/IP Error")
            print(e)
        if returnSuccess:
            return sendResult
        self.connectionThread = None
        
    def _sendCommand(self, cmd):
        result = self._sendTCPIP(cmd, True)
        while result is False:
            result = self._sendTCPIP(cmd, True)
        self.commandThread = None
           

    def sendCommand(self, cmd):
        if self.commandThread is None:
            self.commandThread  = threading.Thread(target = self._sendCommand, args=(cmd,))
            self.commandThread.start()      
            
    def release(self):
        if self.commandThread is not None:
            self.commandThread.join(1)
            self.commandThread = None
        if self.connectionThread is not None:
            self.connectionThread.join(1)
            self.connectionThread = None

#import pandas as pd
#from matplotlib import pyplot as plt


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Connecting to Server

# Connect to PC Server
"""
print(">>> Connecting PC Server ...")

HOST = "192.168.0.37" 
PORT = 8081
data_sent = b"00300000000"

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        s.connect((HOST, PORT))
        s.sendall(data_sent)
        block_ack = s.recv(1024)
        str_block_ack = block_ack.decode('utf-8')
        #print("The data was successfully sent:" + str_block_ack)
    except:
        print("The data has not send yet.")


print(">>> Waiting for PC Server Reply ...")

HOST = "192.168.0.37"
PORT = 8081
BUFFER_SIZE = 1024
#MESSAGE = b"0601"

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
#s.send(MESSAGE)
data = s.recv(BUFFER_SIZE)
s.close()

print ("received data:", data)
print(">>> Start Test ...")
"""


# Connect to iPad Server
""" 
HOST = "127.0.0.1"          #"192.168.0.37"
PORT = 8085
BUFFER_SIZE = 1024
#MESSAGE = b"0601"

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
#s.send(MESSAGE)
data_person = s.recv(BUFFER_SIZE)
s.close()

print ("received data:", str(data_person))

print(">>> Start Test ...")
"""

print(">>> Waiting for iPad Server Reply ...")
"""
HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 8085  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            data_person = conn.recv(1024)
            if not data_person:
                break
            conn.sendall(data_person)
            str_data_person = data_person.decode('utf-8')
            print(str_data_person)
"""

# Parsing Data TEST
str_data_person = "PERSON,50,M"

list_data_person = list(str_data_person.split(","))
nick_name       = list_data_person[0]
age_range_code  = list_data_person[1]
gender_code     = list_data_person[2]



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Connect to Database Server
""" """  

# MySQL Setup
import mysql.connector
from mysql.connector import Error

try:
    mydb = mysql.connector.connect(
        host = "192.168.0.41",                         # Please change to IP Address of server 
        user = "root",                                  # kubotalab                     
        password = "mobimobi",
        database = "sushitech_demo"                      
    )

    mycursor = mydb.cursor()
    sql = "SET time_zone = '+09:00';"
    mycursor.execute(sql)
    mydb.commit()
    print("Time has been set to UTC+9:00")

except Error as e:
    print("Error:", e)
    mydb.rollback()

                                 


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Learning

import torch

# Device configuration
print("GPU CUDA is Available : " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Model Yolo Object Detection
PATH_MODEL = r"C:\Users\50575\Desktop\BDT\MODEL\exp7\weights\best.pt"
model_yolo = torch.hub.load(r"C:\Users\50575\Desktop\BDT\MODEL\yolov5", 'custom', path=PATH_MODEL, force_reload=True, source='local')
model_yolo.to(device)


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HAND TRACKING : MEDIAPIPE

### HAND TRACKING: SETUP
"""
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

joint_list_0 = [[2,1,0], [3,2,1], [4,3,2]]
joint_list_1 = [[6,5,0], [7,6,5], [8,7,6]]
joint_list_2 = [[10,9,0], [11,10,9], [12,11,10]]
joint_list_3 = [[14,13,0], [15,14,13], [16,15,14]]
joint_list_4 = [[18,17,0], [19,18,17], [20,19,18]]

### HAND TRACKING: FUCTION 

def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            
            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [640,480]).astype(int))
            
            output = text, coords
            
    return output



def draw_finger_angles(image, results, joint_list):
    
    # Loop through hands
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 
        for joint in joint_list:

            point = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])
            
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y, hand.landmark[joint[0]].z]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y, hand.landmark[joint[1]].z]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y, hand.landmark[joint[2]].z]) # Third coord

            vector_A = np.array( [ a[0]-b[0], a[1]-b[1] , a[2]-b[2] ])
            vector_B = np.array( [ c[0]-b[0], c[1]-b[1] , c[2]-b[2] ])

            length_A = math.sqrt( pow(a[0]-b[0],2) + pow(a[1]-b[1],2) + pow(a[2]-b[2],2) )
            length_B = math.sqrt( pow(c[0]-b[0],2) + pow(c[1]-b[1],2) + pow(c[2]-b[2],2) )      

            radians = math.acos( np.dot(vector_A, vector_B) / (length_A * length_B) )
            angle = np.abs(radians*180.0/np.pi)
            
            if angle > 180.0:
                angle = 360-angle
                
            #cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(point, [1920, 1080]).astype(int)),
            #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
    return image



def get_finger_angles(results, joint_list):
    
    finger_angles=[]

    # Loop through hands
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 
        
        joint_no = 1
        for joint in joint_list:

            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y, hand.landmark[joint[0]].z]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y, hand.landmark[joint[1]].z]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y, hand.landmark[joint[2]].z]) # Third coord
            
            vector_A = np.array( [ a[0]-b[0], a[1]-b[1] , a[2]-b[2] ])
            vector_B = np.array( [ c[0]-b[0], c[1]-b[1] , c[2]-b[2] ])

            length_A = math.sqrt( pow(a[0]-b[0],2) + pow(a[1]-b[1],2) + pow(a[2]-b[2],2) )
            length_B = math.sqrt( pow(c[0]-b[0],2) + pow(c[1]-b[1],2) + pow(c[2]-b[2],2) )      

            radians = math.acos( np.dot(vector_A, vector_B) / (length_A * length_B) )
            angle = np.abs(radians*180.0/np.pi)
            
            #if joint_no == 1 and angle < 90 :
            #    angle = 90
            #elif joint_no == 2 and angle < 110 :
            #    angle = 110
            #elif joint_no == 3 and angle < 90 :
            #    angle = 90
            
            joint_no = joint_no + 1
            finger_angles.append(round(angle, 2))

    return finger_angles
"""


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Read Files

task_00 = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\00.jpg')
task_0A = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\0A.jpg')
task_0B = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\0B.jpg')
task_01 = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\01.jpg')
task_02 = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\02.jpg')
task_03 = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\03.jpg')
task_04 = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\04.jpg')
task_05 = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\05.jpg')
task_06 = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\06.jpg')
task_07 = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\07.jpg')
task_08 = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\08.jpg')
task_09 = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\09.jpg')

face_01 = cv2.imread(r"C:\Users\50575\Desktop\BDT\FILES\LABEL_50x50\01.png", cv2.IMREAD_COLOR)
face_02 = cv2.imread(r"C:\Users\50575\Desktop\BDT\FILES\LABEL_50x50\02.png", cv2.IMREAD_COLOR)
face_03 = cv2.imread(r"C:\Users\50575\Desktop\BDT\FILES\LABEL_50x50\03.png", cv2.IMREAD_COLOR)
face_04 = cv2.imread(r"C:\Users\50575\Desktop\BDT\FILES\LABEL_50x50\04.png", cv2.IMREAD_COLOR)
face_05 = cv2.imread(r"C:\Users\50575\Desktop\BDT\FILES\LABEL_50x50\05.png", cv2.IMREAD_COLOR)
face_06 = cv2.imread(r"C:\Users\50575\Desktop\BDT\FILES\LABEL_50x50\06.png", cv2.IMREAD_COLOR)

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

img_00 = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_100x100\00.jpg')
img_01 = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_100x100\01.jpg')
img_02 = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_100x100\02.jpg')
img_03 = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_100x100\03.jpg')
img_04 = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_100x100\04.jpg')
img_05 = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_100x100\05.jpg')
img_06 = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_100x100\06.jpg')
img_07 = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_100x100\07.jpg')
img_08 = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_100x100\08.jpg')
img_09 = cv2.imread(r'C:\Users\50575\Desktop\BDT\FILES\TEST_100x100\09.jpg')

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



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Text to Speech 

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



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SEtup GUI 

import customtkinter
from PIL import Image, ImageTk
import cv2

customtkinter.set_appearance_mode("System")         # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

#PATH_VIDEO = r"C:\Users\50575\Desktop\BDT\VIDEO_EXPERIMENT\TLL\TOP_VIEW\anom_02.mp4"
#cap = cv2.VideoCapture(PATH_VIDEO)          #0, cv2.CAP_DSHOW

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)      #1280   640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)     #720    480



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
        IMAGE_PATH = r"C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\0B.jpg"
        self.image = customtkinter.CTkImage(light_image=Image.open(os.path.join(IMAGE_PATH)), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
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
        self.button_0.grid(row=2, column=0, padx=65, pady=65, sticky="nswe") # , columnspan=2, row=2

        self.button_1 = customtkinter.CTkButton(self, text="アプリ終了 (EXIT)", font=("Helvetica", 30), command=self.button_1_callback)
        self.button_1.grid(row=2, column=1, padx=65, pady=65, sticky="nswe") # , columnspan=2, row=2

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

        self.cognitive_age_list = []

        global nick_name
        global gender_code
        global age_range_code

        self.nick_name = nick_name
        self.gender_code = gender_code
        self.age_range_code = age_range_code

        #self.task_state = 1

        SpeakText("ブロックデザインゲームえようこそ。")
        SpeakText("赤と白の 4 つのブロックから表示されたデザインを作成してください。")
        #SpeakText("8 つの問題を解くのにかかる平均時間を計測します。")
        SpeakText("左手または利き手と反対の手で操作してください。")
        SpeakText("準備ができたらスタートボタンを押してください。")

        
        #self.conn = ServerConnection()
        #self.conn.getUDPServer()
        #self.conn.sendCommand("00300000000")

        # self.conn.release()
        # print(">>> Connecting Server ...")
        # while(1):
            
        #     HOST = "192.168.0.55"       # "192.168.0.37" 
        #     PORT = 9866                 # 8081
        #     data_sent = b"00300000000"  #b"00317020617" 

        #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        #         try:
        #             s.connect((HOST, PORT))
        #             s.sendall(data_sent)
        #             block_ack = s.recv(1024)
        #             str_block_ack = block_ack.decode('utf-8')
        #             #print("The data was successfully sent:" + str_block_ack)
        #         except:
        #             print("The data has not send yet.")
        #     s.close()


        # print(">>> Waiting for Reply ...")

        # HOST = "192.168.0.55"       # "192.168.0.37" 
        # PORT = 9866                 # 8081
        # BUFFER_SIZE = 1024
        # #MESSAGE = b"0601"

        # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # s.connect((HOST, PORT))
        # #s.send(MESSAGE)
        # data = s.recv(BUFFER_SIZE)
        # s.close()

        # print ("received data:", data)

        # print(">>> Start Test ...")
        # """  """

    

    def button_0_callback(self):

        SpeakText("3")
        SpeakText("2")
        SpeakText("1")
        SpeakText("Start！")
        print("START")

        IMAGE_WIDTH = 525 #650
        IMAGE_HEIGHT = 525 #650
        IMAGE_PATH = r"C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\01.jpg"
        self.image = customtkinter.CTkImage(light_image=Image.open(os.path.join(IMAGE_PATH)), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
        self.label = customtkinter.CTkLabel(master=self.design_frame_1, image=self.image, text='')
        self.label.grid(column=0, row=0)

        self.start_task = time.time()
        self.state = "disabled"
        self.streaming()




    def button_1_callback(self):

        #os.execl(sys.executable, os.path.abspath(__file__), *sys.argv)
        sys.exit() 



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
        
        return int(cognitive_age) #.tolist()[0]

        #print("Estimated Cognitive Age : " + str(round(cognitive_age[0], 1)) + " years")



    # code for video streaming
    def streaming(self):

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
                
                #print("----- TASK :" + str(box_design_sort))

                if box_design_sort == ans_01:
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
                        SpeakText("問題 1 は  " + str(int(self.timer_task_01)) +"  秒できました。")
                        SpeakText("次は問題 2 です。")

                        #IMAGE_WIDTH = 650
                        #IMAGE_HEIGHT = 650
                        IMAGE_PATH = r"C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\02.jpg"
                        self.image = customtkinter.CTkImage(light_image=Image.open(os.path.join(IMAGE_PATH)), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
                        self.label = customtkinter.CTkLabel(master=self.design_frame_1, image=self.image, text='')
                        self.label.grid(column=0, row=0)

                        self.task_flag_01 = False
                    #else:
                        #print ("TASK 1 COMPLETED in " + str(self.timer_task_01) +" seconds")
                    self.start_task = time.time()
                    
                    # Detect Thumb
                    """
                    if (thumb_state == 1 and thumb_flag_01 == True):
                        end_thumb = time.time()
                        timer_thumb = end_thumb - start_thumb
                        t_thumb_01 = round((timer_thumb - self.timer_task_01), 2)
                        t_thumb_all.append(t_thumb_01)

                        start_thumb = time.time()
                        thumb_flag_01 = False

                        SpeakText("Good for a start.")
                    """


                elif box_design_sort == ans_02 or box_design_sort == ans_02x:
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
                        SpeakText("問題 2 は  " + str(int(self.timer_task_02)) +"  秒できました。")
                        SpeakText("次は問題 3 です。")

                        #IMAGE_WIDTH = 650
                        #IMAGE_HEIGHT = 650
                        IMAGE_PATH = r"C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\03.jpg"
                        self.image = customtkinter.CTkImage(light_image=Image.open(os.path.join(IMAGE_PATH)), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
                        self.label = customtkinter.CTkLabel(master=self.design_frame_1, image=self.image, text='')
                        self.label.grid(column=0, row=0)

                        self.task_flag_02 = False
                    #else:
                        #print ("TASK 2 COMPLETED in " + str(self.timer_task_02) +" seconds")
                    self.start_task = time.time()

                    # Detect Thumb
                    """
                    if (thumb_state == 1 and thumb_flag_02 == True):
                        end_thumb = time.time()
                        timer_thumb = end_thumb - start_thumb
                        t_thumb_02 = round((timer_thumb - self.timer_task_02), 2)
                        t_thumb_all.append(t_thumb_02)

                        start_thumb = time.time()
                        thumb_flag_02 = False

                        timer_speak = self.timer_task_02
                        if timer_speak < 10:
                            SpeakText("Amazing.")
                        if timer_speak < 15:
                            SpeakText("Awesome.")
                        elif timer_speak < 20:
                            SpeakText("Great.")
                        elif timer_speak < 25:
                            SpeakText("Good job.")
                        elif timer_speak < 30:
                            SpeakText("Well done.")
                        else:
                            SpeakText("Give your best.")
                    """


                elif box_design_sort == ans_03 or box_design_sort == ans_03x:
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
                        SpeakText("問題 3 は  " + str(int(self.timer_task_03)) +" 秒できました。")
                        SpeakText("次は問題 4 です。")

                        #IMAGE_WIDTH = 650
                        #IMAGE_HEIGHT = 650
                        IMAGE_PATH = r"C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\04.jpg"
                        self.image = customtkinter.CTkImage(light_image=Image.open(os.path.join(IMAGE_PATH)), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
                        self.label = customtkinter.CTkLabel(master=self.design_frame_1, image=self.image, text='')
                        self.label.grid(column=0, row=0)                           

                        self.task_flag_03 = False
                    #else:
                        #print ("TASK 3 COMPLETED in " + str(self.timer_task_03) +" seconds")
                    self.start_task = time.time()
                    
                    # Detect Thumb
                    """
                    if (thumb_state == 1 and thumb_flag_03 == True):
                        end_thumb = time.time()
                        timer_thumb = end_thumb - start_thumb
                        t_thumb_03 = round((timer_thumb - self.timer_task_03), 2)
                        t_thumb_all.append(t_thumb_03)

                        start_thumb = time.time()
                        thumb_flag_03 = False

                        timer_speak = self.timer_task_03
                        if timer_speak < 10:
                            SpeakText("Amazing.")
                        if timer_speak < 15:
                            SpeakText("Awesome.")
                        elif timer_speak < 20:
                            SpeakText("Great.")
                        elif timer_speak < 25:
                            SpeakText("Good job.")
                        elif timer_speak < 30:
                            SpeakText("Well done.")
                        else:
                            SpeakText("Give your best.")
                    """

                                    
                elif box_design_sort == ans_04 or box_design_sort == ans_04x:
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
                        SpeakText("問題 4 は  " + str(int(self.timer_task_04)) +" 秒できました。")
                        SpeakText("次は問題 5 です。")

                        #IMAGE_WIDTH = 650
                        #IMAGE_HEIGHT = 650
                        IMAGE_PATH = r"C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\05.jpg"
                        self.image = customtkinter.CTkImage(light_image=Image.open(os.path.join(IMAGE_PATH)), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
                        self.label = customtkinter.CTkLabel(master=self.design_frame_1, image=self.image, text='')
                        self.label.grid(column=0, row=0)

                        self.task_flag_04 = False
                    #else:
                        #print ("TASK 4 COMPLETED in " + str(self.timer_task_04) +" seconds")
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
                            SpeakText("Amazing.")
                        if timer_speak < 15:
                            SpeakText("Awesome.")
                        elif timer_speak < 20:
                            SpeakText("Great.")
                        elif timer_speak < 25:
                            SpeakText("Good job.")
                        elif timer_speak < 30:
                            SpeakText("Well done.")
                        else:
                            SpeakText("Give your best.")
                    """


                elif box_design_sort == ans_05 or box_design_sort == ans_05x:
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
                        SpeakText("問題 5 は  " + str(int(self.timer_task_05)) +" 秒できました。")
                        SpeakText("次は問題 6 です。")

                        #IMAGE_WIDTH = 650
                        #IMAGE_HEIGHT = 650
                        IMAGE_PATH = r"C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\06.jpg"
                        self.image = customtkinter.CTkImage(light_image=Image.open(os.path.join(IMAGE_PATH)), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
                        self.label = customtkinter.CTkLabel(master=self.design_frame_1, image=self.image, text='')
                        self.label.grid(column=0, row=0)

                        self.task_flag_05 = False
                    #else:
                        #print ("TASK 5 COMPLETED in " + str(self.timer_task_05) +" seconds")
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
                            SpeakText("Amazing.")
                        if timer_speak < 15:
                            SpeakText("Awesome.")
                        elif timer_speak < 20:
                            SpeakText("Great.")
                        elif timer_speak < 25:
                            SpeakText("Good job.")
                        elif timer_speak < 30:
                            SpeakText("Well done.")
                        else:
                            SpeakText("Give your best.")
                    """


                elif box_design_sort == ans_06 or box_design_sort == ans_06x:
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
                        SpeakText("問題 6 は  " + str(int(self.timer_task_06)) +" 秒できました。")
                        SpeakText("次は問題 7 です。")

                        #IMAGE_WIDTH = 650
                        #IMAGE_HEIGHT = 650
                        IMAGE_PATH = r"C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\07.jpg"
                        self.image = customtkinter.CTkImage(light_image=Image.open(os.path.join(IMAGE_PATH)), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
                        self.label = customtkinter.CTkLabel(master=self.design_frame_1, image=self.image, text='')
                        self.label.grid(column=0, row=0)

                        self.task_flag_06 = False
                    #else:
                        #print ("TASK 6 COMPLETED in " + str(self.timer_task_06) +" seconds")
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
                            SpeakText("Amazing.")
                        if timer_speak < 15:
                            SpeakText("Awesome.")
                        elif timer_speak < 20:
                            SpeakText("Great.")
                        elif timer_speak < 25:
                            SpeakText("Good job.")
                        elif timer_speak < 30:
                            SpeakText("Well done.")
                        else:
                            SpeakText("Give your best.")
                    """


                elif box_design_sort == ans_07 or box_design_sort == ans_07x:
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
                        SpeakText("問題 7 は  " + str(int(self.timer_task_07)) +" 秒できました。")
                        SpeakText("次は問題 8 です。")

                        #IMAGE_WIDTH = 650
                        #IMAGE_HEIGHT = 650
                        IMAGE_PATH = r"C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\08.jpg"
                        self.image = customtkinter.CTkImage(light_image=Image.open(os.path.join(IMAGE_PATH)), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
                        self.label = customtkinter.CTkLabel(master=self.design_frame_1, image=self.image, text='')
                        self.label.grid(column=0, row=0)

                        self.task_flag_07 = False
                    #else:
                        #print ("TASK 7 COMPLETED in " + str(self.timer_task_07) +" seconds")
                    self.start_task = time.time()
                    
                    # Detect Thumb
                    """
                    if (thumb_state == 1 and thumb_flag_07 == True):
                        end_thumb = time.time()
                        timer_thumb = end_thumb - start_thumb
                        t_thumb_07 = round((timer_thumb - self.timer_task_07), 2)
                        t_thumb_all.append(t_thumb_07)

                        start_thumb = time.time()
                        thumb_flag_07 = False

                        timer_speak = self.timer_task_07
                        if timer_speak < 10:
                            SpeakText("Amazing.")
                        if timer_speak < 15:
                            SpeakText("Awesome.")
                        elif timer_speak < 20:
                            SpeakText("Great.")
                        elif timer_speak < 25:
                            SpeakText("Good job.")
                        elif timer_speak < 30:
                            SpeakText("Well done.")
                        else:
                            SpeakText("Give your best.")
                    """


                elif box_design_sort == ans_08 or box_design_sort == ans_08x:
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
                        SpeakText("問題 8 は " + str(int(self.timer_task_08)) +" 秒できました。")
                        SpeakText("これで、ブロックデザインゲームは終了です。お疲れさまでした。")
                        
                        self.timer_task_avg = int(sum(self.timer_task_all) / len(self.timer_task_all))
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

                        print("Your average Completed Time is " + str( self.timer_task_avg) +" second")
                        print("Your average Estimated Cognitive Age is " + str( cognitive_age_avg) +" years")

                        # Data transfered from the iPad Controller


                        self.nick_name 			= "'" + str( self.nick_name)       + "'" 		    # nick_name
                        self.gender_code 		= "'" + str( self.gender_code)		+ "'" 		# male
                        self.age_range_code		= "'" + str( self.age_range_code)	+ "'" 		    # class
                        time_average			= "'" + str( self.timer_task_avg) + "'"      # seconds
                        estimate_cognitive_age 	= "'" + str( cognitive_age_avg) + "'"        # years

                        # ------------------
                        """
                        print(">>> Connecting Server ...")

                        HOST = "192.168.0.37" 
                        PORT = 8081
                        data_sent = b"00317020500"

                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            try:
                                s.connect((HOST, PORT))
                                s.sendall(data_sent)
                                block_ack = s.recv(1024)
                                str_block_ack = block_ack.decode('utf-8')
                                #print("The data was successfully sent:" + str_block_ack)
                            except:
                                print("The data has not send yet.")
                        """

                        # ------------------
                        """  """
                        try : 	
                            sql = "INSERT INTO block_design_test (date, time, nick_name, gender_code, age_range_code, time_average, estimate_cognitive_age) VALUES (curdate(),curtime(),{},{},{},{},{})".format(self.nick_name, self.gender_code, self.age_range_code, time_average, estimate_cognitive_age)

                            mycursor.execute(sql)
                            mydb.commit()

                        except Error as e:
                            print("Error:", e)
                            mydb.rollback()


                        # finally:
                        if (mydb.is_connected()):
                            mydb.close()
                            mycursor.close()
                            print("MySQL connection is closed.")
                        
                        

                        SpeakText("平均完了時間は " + str( self.timer_task_avg) +" 秒です")
                        SpeakText("あなたの推定認知年齢は " + str( cognitive_age_avg) +" 歳です")

                        #if cognitive_age_avg <= 30:
                        #    SpeakText("あなたの推定認知年齢は 30 歳未満です")
                        #elif cognitive_age_avg > 30 and cognitive_age_avg <= 40 :
                        #    SpeakText("あなたの推定認知年齢は 40 歳から 50 歳の間です")
                        #elif cognitive_age_avg > 40 and cognitive_age_avg <= 50 :
                        #    SpeakText("あなたの推定認知年齢は 40 歳から 50 歳の間です")
                        #elif cognitive_age_avg > 50 and cognitive_age_avg <= 60 :
                        #    SpeakText("あなたの推定認知年齢は 50 歳から 60 歳の間です")
                        #elif cognitive_age_avg > 60 and cognitive_age_avg <= 70 :
                        #    SpeakText("あなたの推定認知年齢は 60 歳から 70 歳の間です")
                        #elif cognitive_age_avg > 70 :
                        #    SpeakText("あなたの推定認知年齢は 70 歳を超えています")

                        #SpeakText("ブロックゲームにご参加いただきありがとうございます。")
                        #SpeakText("また会いましょう、さようなら。")
                        
                        #>>>>>>>>>>>>>>>>>>>>>>

                        #IMAGE_WIDTH = 650
                        #IMAGE_HEIGHT = 650
                        IMAGE_PATH = r"C:\Users\50575\Desktop\BDT\FILES\TEST_1000x1000\0B.jpg"
                        self.image = customtkinter.CTkImage(light_image=Image.open(os.path.join(IMAGE_PATH)), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
                        self.label = customtkinter.CTkLabel(master=self.design_frame_1, image=self.image, text='')
                        self.label.grid(column=0, row=0)

                        self.task_flag_08 = False

                        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        #break
                    
                    #else:
                        #print ("TASK 8 COMPLETED in " + str(self.timer_task_08) +" seconds")
                    self.start_task = time.time()

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
                            SpeakText("Congratulations you did it !")
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
