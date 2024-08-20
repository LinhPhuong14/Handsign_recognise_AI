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
