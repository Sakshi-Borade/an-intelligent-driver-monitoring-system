from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import PhotoImage
from pygame import mixer

mixer.init()
mixer.music.load("buzzer.wav")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_drowsiness():
    thresh = 0.25
    frame_check = 30
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    cap = cv2.VideoCapture(0)
    flag = 0

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    cv2.putText(frame, "*****ALERT!*****", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "*****ALERT!*****", (10, 325),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    mixer.music.play()
            else:
                flag = 0

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()

def start_detection():
    messagebox.showinfo("Start Detection", "Drowsiness detection will start now.")
    detect_drowsiness()

root = tk.Tk()
root.geometry("700x500")
bg = PhotoImage(file = "mega project bg.png") 
root.title("Driver Drowsiness Detection System")

label = tk.Label(root, text="Driver  Drowsiness", font=("Lucida Handwriting", 30),fg="#fdd017",bg="#2f2e2e")
label.place(x=140,y=100)
label1 = tk.Label(root, text="Detection  System", font=("Lucida Handwriting", 30),fg="#fdd017",bg="#2f2e2e")
label1.place(x=150,y=150)


start_button = tk.Button(root, text="Start  Detection",font=("Lucida Handwriting", 16), fg="#fdd017",bg="black", command=start_detection)
start_button.place(x=240,y=300)

root.mainloop()