import tkinter as tk
from tkinter import messagebox
import cv2
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
from pygame import mixer

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_drowsiness():
    COUNTER = 0
    ALARM_ON = False

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            if ear < 0.25:
                COUNTER += 1
                if COUNTER >= 30:
                    if not ALARM_ON:
                        ALARM_ON = True
                        messagebox.showwarning("Warning", "Driver Drowsiness Detected!")
            else:
                COUNTER = 0
                ALARM_ON = False

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

root = tk.Tk()
root.title("Driver Drowsiness Detection System")

label = tk.Label(root, text="Driver Drowsiness Detection System", font=("Helvetica", 18))
label.pack(pady=20)

start_button = tk.Button(root, text="Start Detection", command=detect_drowsiness)
start_button.pack()

root.mainloop()