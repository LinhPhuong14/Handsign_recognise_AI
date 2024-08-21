import mediapipe as mp
import joblib
import numpy as np
import cv2
from timeit import default_timer as timer
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
  model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands = 1
) 
def eulidean_distance(landmarkA, landmarkB):
    A = np.array([landmarkA.x, landmarkA.y])
    B = np.array([landmarkB.x, landmarkB.y])
    distance = np.linalg.norm(A-B)
    return distance

def process(image, live=True):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    data = []
    footage = np.zeros(image.shape)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark 
        data = add_distance(landmarks)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                footage,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    else:
        data = [0 for i in range(0, 16)]
    if live:
        cv2.imshow('MediaPipe Hands', cv2.flip(footage, 1))
    return data

characters = ['A',
 'B',
 'C',
 'D',
 'E',
 'F',
 'G',
 'H',
 'I',
 'K',
 'L',
 'M',
 'N',
 'O',
 'P',
 'Q',
 'R',
 'S',
 'T',
 'U',
 'V',
 'W',
 'X',
 'Y',
 'delete',
 'space'
 ]

def add_distance(landmarks):
    data = []
    data.append(eulidean_distance(landmarks[4], landmarks[0]))
    data.append(eulidean_distance(landmarks[8], landmarks[0]))
    data.append(eulidean_distance(landmarks[12], landmarks[0]))
    data.append(eulidean_distance(landmarks[16], landmarks[0]))
    data.append(eulidean_distance(landmarks[20], landmarks[0]))
    data.append(eulidean_distance(landmarks[4], landmarks[8]))
    data.append(eulidean_distance(landmarks[4], landmarks[12]))
    data.append(eulidean_distance(landmarks[8], landmarks[12]))
    data.append(eulidean_distance(landmarks[12], landmarks[16]))
    data.append(eulidean_distance(landmarks[20], landmarks[16]))
    data.append(eulidean_distance(landmarks[8], landmarks[16]))
    data.append(eulidean_distance(landmarks[8], landmarks[20]))
    data.append(eulidean_distance(landmarks[12], landmarks[20]))
    data.append(eulidean_distance(landmarks[4], landmarks[12]))
    data.append(eulidean_distance(landmarks[4], landmarks[16]))
    data.append(eulidean_distance(landmarks[4], landmarks[20]))
    return data
svm_model = joblib.load('svm_model.pkl')

cap = cv2.VideoCapture(0)
count = 0
while True:
    start = timer()
    ret, frame = cap.read()
    if not ret:
        print("Cannot read video stream")
        break
    data = np.array(process(frame)) 
    if np.all(data == 0):
        print("No hand detected")
    else:
        result = svm_model.predict_proba(data.reshape(1, -1))[0]
        classes = np.argmax(result)
        prob = result[classes]
        print(characters[classes], end="")
    stop = timer()
    print(f' - Execution time: {"{:.4f}".format((stop-start)*1000)} ms - FPS: {int(1/(stop-start))}')
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
