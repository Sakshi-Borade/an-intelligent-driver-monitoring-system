import tkinter as tk
from tkinter import messagebox
import os

def start_application():
    root.destroy()
    os.system("Drowsiness_Detection.py")

root = tk.Tk()
root.title("Driver Drowsiness Detection System")

label = tk.Label(root, text="Welcome to Driver Drowsiness Detection System", font=("Helvetica", 18))
label.pack(pady=20)

start_button = tk.Button(root, text="Start Application", command=start_application)
start_button.pack()

root.mainloop()