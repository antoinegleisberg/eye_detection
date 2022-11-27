import cv2
import mediapipe as mp
import numpy as np

from landmark_detector import LandmarkDetector

# mediapipe modules
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


class EyeDetector:
    def __init__(self):
        self.landmark_detector = LandmarkDetector()
        self.videoCapture = cv2.VideoCapture(0)
        self.landmark = None
        self.right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.right_iris_indices = [469, 470, 471, 472]
        self.left_iris_indices = [474, 475, 476, 477]
        self.right_eyebrow_indices = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
        self.left_eyebrow_indices = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]

    def get_frame_landmarks(self):
        success, image = self.videoCapture.read()
        if not success:
            return
        self.landmark_detector.set_image(image)
        results = self.landmark_detector.process_image()
        if results.multi_face_landmarks:
            self.landmark = results.multi_face_landmarks[0].landmark

    def run(self):
        while self.videoCapture.isOpened():
            image = np.zeros((480, 640, 3), np.uint8)
            self.get_frame_landmarks()
            for idx in self.right_eye_indices + self.left_eye_indices:
                landmark = self.landmark[idx]
                image = cv2.circle(image, (int(landmark.x * 640), int(landmark.y * 480)), 2, (0, 255, 0), -1)
            for idx in self.right_iris_indices + self.left_iris_indices:
                landmark = self.landmark[idx]
                image = cv2.circle(image, (int(landmark.x * 640), int(landmark.y * 480)), 2, (0, 0, 255), -1)
            for idx in self.right_eyebrow_indices + self.left_eyebrow_indices:
                landmark = self.landmark[idx]
                image = cv2.circle(image, (int(landmark.x * 640), int(landmark.y * 480)), 2, (255, 0, 0), -1)
            image = cv2.flip(image, 1)
            cv2.imshow("image", image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.videoCapture.release()


if __name__ == "__main__":
    eye_detector = EyeDetector()
    eye_detector.run()
