import cv2
import mediapipe as mp
import numpy as np
from enum import Enum
from utils.coordinates import Coordinates

from landmark_detector import LandmarkDetector

# mediapipe modules
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


class Direction(Enum):
    TOP_LEFT = (-1, -1)
    TOP_RIGHT = (1, -1)
    BOTTOM_LEFT = (-1, 1)
    BOTTOM_RIGHT = (1, 1)
    CENTER_LEFT = (-1, 0)
    CENTER_RIGHT = (1, 0)
    TOP_CENTER = (0, -1)
    BOTTOM_CENTER = (0, 1)
    CENTER = (0, 0)


class EyeDetector:
    def __init__(self):
        self.landmark_detector = LandmarkDetector()
        self.videoCapture = cv2.VideoCapture(0)
        self.landmark = None
        self.landmark_indices = {
            "right_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            "left_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            "right_iris": [469, 470, 471, 472],
            "left_iris": [474, 475, 476, 477],
            "right_eyebrow": [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
            "left_eyebrow": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
        }
        self.reference_position = None

    def get_frame_landmarks(self) -> None:
        success, image = self.videoCapture.read()
        if not success:
            return
        self.landmark_detector.set_image(image)
        results = self.landmark_detector.process_image()
        if results.multi_face_landmarks:
            self.landmark = results.multi_face_landmarks[0].landmark

    def draw_landmarks(self) -> None:
        image = np.zeros((480, 640, 3), np.uint8)
        for idx in self.landmark_indices["right_eye"]:
            landmark = self.landmark[idx]
            image = cv2.circle(image, (int(landmark.x * 640), int(landmark.y * 480)), 2, (0, 255, 0), -1)
        for idx in self.landmark_indices["left_eye"]:
            landmark = self.landmark[idx]
            image = cv2.circle(image, (int(landmark.x * 640), int(landmark.y * 480)), 2, (0, 255, 255), -1)
        for idx in self.landmark_indices["right_iris"]:
            landmark = self.landmark[idx]
            image = cv2.circle(image, (int(landmark.x * 640), int(landmark.y * 480)), 2, (0, 0, 255), -1)
        for idx in self.landmark_indices["left_iris"]:
            landmark = self.landmark[idx]
            image = cv2.circle(image, (int(landmark.x * 640), int(landmark.y * 480)), 2, (255, 255, 0), -1)
        for idx in self.landmark_indices["right_eyebrow"]:
            landmark = self.landmark[idx]
            image = cv2.circle(image, (int(landmark.x * 640), int(landmark.y * 480)), 2, (255, 0, 0), -1)
        for idx in self.landmark_indices["left_eyebrow"]:
            landmark = self.landmark[idx]
            image = cv2.circle(image, (int(landmark.x * 640), int(landmark.y * 480)), 2, (255, 0, 255), -1)
        image = cv2.flip(image, 1)
        cv2.imshow("image", image)

    def calibrate(self) -> None:
        image = np.zeros((480, 640, 3), np.uint8)
        image = cv2.putText(
            image,
            "Please look at the center of the screen and press any key to continue",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("image", image)
        cv2.waitKey(0)
        self.get_frame_landmarks()
        self.reference_position = {
            "landmarks": self.landmark,
            "eyes_coords": self.eyes_coords,
            "normal": self.normal,
        }

    @property
    def normal(self) -> Coordinates:
        return

    @property
    def eyes_coords(self):
        # Get landmarks coords
        right_eye_coords = {
            idx: Coordinates(self.landmark[idx].x, self.landmark[idx].y) for idx in self.landmark_indices["right_eye"]
        }
        left_eye_coords = {
            idx: Coordinates(self.landmark[idx].x, self.landmark[idx].y) for idx in self.landmark_indices["left_eye"]
        }
        right_iris_coords = {
            idx: Coordinates(self.landmark[idx].x, self.landmark[idx].y) for idx in self.landmark_indices["right_iris"]
        }
        left_iris_coords = {
            idx: Coordinates(self.landmark[idx].x, self.landmark[idx].y) for idx in self.landmark_indices["left_iris"]
        }

        # Normalize coords
        right_eye = Coordinates.normalize_dict({**right_eye_coords, **right_iris_coords})
        normalized_right_eye_coords = {idx: right_eye[idx] for idx in self.landmark_indices["right_eye"]}
        normalized_right_iris_coords = {idx: right_eye[idx] for idx in self.landmark_indices["right_iris"]}
        left_eye = Coordinates.normalize_dict({**left_eye_coords, **left_iris_coords})
        normalized_left_eye_coords = {idx: left_eye[idx] for idx in self.landmark_indices["left_eye"]}
        normalized_left_iris_coords = {idx: left_eye[idx] for idx in self.landmark_indices["left_iris"]}

        return {
            "right_eye": right_eye_coords,
            "left_eye": left_eye_coords,
            "right_iris": right_iris_coords,
            "left_iris": left_iris_coords,
            "normalized_right_eye": normalized_right_eye_coords,
            "normalized_left_eye": normalized_left_eye_coords,
            "normalized_right_iris": normalized_right_iris_coords,
            "normalized_left_iris": normalized_left_iris_coords,
        }

    @property
    def looking_at(self) -> Direction:
        if self.landmark is None:
            return Direction.CENTER

        # Calculate centers
        eye_coords = self.eye_coords
        right_iris_coords = eye_coords["right_iris"]
        left_iris_coords = eye_coords["left_iris"]
        right_iris_center = sum([coord for coord in right_iris_coords.values()], Coordinates(0, 0)) / len(
            right_iris_coords
        )
        left_iris_center = sum([coord for coord in left_iris_coords.values()], Coordinates(0, 0)) / len(
            left_iris_coords
        )

        # Get direction
        # print(right_eye_center, left_eye_center, right_iris_center, left_iris_center)
        if right_iris_center.x > 0.55 and left_iris_center.x > 0.55:
            return Direction.CENTER_LEFT
        elif right_iris_center.x < 0.45 and left_iris_center.x < 0.45:
            return Direction.CENTER_RIGHT
        return Direction.CENTER

    def run(self) -> None:
        while self.videoCapture.isOpened():
            self.get_frame_landmarks()
            self.draw_landmarks()
            print(self.landmark.z)
            print(self.looking_at)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.videoCapture.release()


if __name__ == "__main__":
    eye_detector = EyeDetector()
    eye_detector.run()
