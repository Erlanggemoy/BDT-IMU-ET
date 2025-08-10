import customtkinter

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



class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("User Input")
        self.geometry("480x320")
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.resizable(0,0)


        self.textbox_frame_0 = MyTextboxFrame(self, "Nick Name", values="")
        self.textbox_frame_0.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 0), sticky="nsew")

        self.textbox_frame_1 = MyTextboxFrame(self, "Age", values="")
        self.textbox_frame_1.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="nsew")      
        
        self.radiobutton_frame_0 = MyRadiobuttonFrame(self, "Gender", values=["male", "female"])
        self.radiobutton_frame_0.grid(row=1, column=1, padx=(0, 10), pady=(10, 0), sticky="nsew")

        self.button = customtkinter.CTkButton(self, text="SUBMIT", command=self.button_callback, font=("Helvetica",30))
        self.button.grid(row=2, column=0, padx=10, pady=10, sticky="ew", columnspan=2)

    def button_callback(self):
        print("Name   : ", self.textbox_frame_0.get())
        print("Age    : ", self.textbox_frame_1.get())
        print("Gender : ", self.radiobutton_frame_0.get())
        
        self.destroy()


app = App()
app.mainloop()