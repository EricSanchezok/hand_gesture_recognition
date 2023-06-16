# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

import mediapipe as mp
import time




mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.8, min_tracking_confidence=0.1)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0,0,255), thickness=5)
handConStyle = mpDraw.DrawingSpec(color=(0,255,0), thickness=5)

pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)

while True:
    frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    imgHeight, imgWidth, _ = frame.shape

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (30, 100), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 255), 5)
                


    cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
    cv2.imshow('Align Example', frame)

    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break

