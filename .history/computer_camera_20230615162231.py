# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

import mediapipe as mp
import time




mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.95, min_tracking_confidence=0.95)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0,0,255), thickness=5)
handConStyle = mpDraw.DrawingSpec(color=(0,255,0), thickness=5)

pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)

while True:
    frames = cap.read()

    color_image = np.asanyarray(frames)

    results = hands.process(cv2.cvtColor(frames, cv2.COLOR_BGR2RGB))

    imgHeight, imgWidth, _ = color_image.shape

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(color_image, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(color_image, str(int(fps)), (30, 100), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 255), 5)
                


    cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
    cv2.imshow('Align Example', color_image)

    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break

    isTrain = True
    #竖大拇指
    if key & 0xFF == ord('0'):
        writeData(results, 0, isTrain)
    #按住大拇指
    if key & 0xFF == ord('1'):
        writeData(results, 1, isTrain)
    #OK
    if key & 0xFF == ord('2'):
        writeData(results, 2, isTrain)
    #五指张开
    if key & 0xFF == ord('3'):
        writeData(results, 3, isTrain)
    #其他
    if key & 0xFF == ord('4'):
        writeData(results, 4, isTrain)
                        
if isTrain:
    #对train_data.csv按照最后一列进行排序
    with open('train_data.csv', 'r') as f:
        lines = f.readlines()
        lines.sort(key=lambda line: int(line.split(',')[-1]))
    with open('train_data.csv', 'w') as f:
        f.writelines(lines)
