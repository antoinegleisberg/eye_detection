import cv2
import mediapipe as mp
import numpy as np
from enum import Enum
from utils.coordinates import Coordinates
from screeninfo import get_monitors
from typing import Mapping

from landmark_detector import LandmarkDetector
from utils.mp_faces import mediapipe_faces

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
    def __init__(self, fullscreen: bool = False):
        self.landmark_detector = LandmarkDetector()
        self.screen_dimensions = Coordinates(get_monitors()[0].width, get_monitors()[0].height)
        self.camera_dimensions = Coordinates(640, 480)  # width, height
        self.videoCapture = cv2.VideoCapture(0)
        self.landmarks = None
        self.landmark_indices = {
            "right_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            "left_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            "right_iris": [469, 470, 471, 472],
            "left_iris": [474, 475, 476, 477],
            "right_eyebrow": [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
            "left_eyebrow": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
        }
        if fullscreen:
            cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.reference = dict()
        self.calibrate()

    def get_frame_landmarks(self) -> bool:
        """Gets the landmarks of the current frame and stores them in self.landmarks"""
        success, image = self.videoCapture.read()
        if not success:
            return False
        self.landmark_detector.set_image(image)
        results = self.landmark_detector.process_image()
        if results.multi_face_landmarks:
            self.landmarks = results.multi_face_landmarks[0].landmark
            return True
        return False

    def draw_landmarks(self) -> None:
        """Draws the landmarks on an empty image and displays it"""
        width, height = self.camera_dimensions.x, self.camera_dimensions.y
        image = np.zeros((height, width, 3), np.uint8)  # numpy first 2 dimensions are switched
        for idx in self.landmark_indices["right_eye"]:
            landmark = self.landmarks[idx]
            image = cv2.circle(image, (int(landmark.x * width), int(landmark.y * height)), 2, (0, 255, 0), -1)
        for idx in self.landmark_indices["left_eye"]:
            landmark = self.landmarks[idx]
            image = cv2.circle(image, (int(landmark.x * width), int(landmark.y * height)), 2, (0, 255, 255), -1)
        for idx in self.landmark_indices["right_iris"]:
            landmark = self.landmarks[idx]
            image = cv2.circle(image, (int(landmark.x * width), int(landmark.y * height)), 2, (0, 0, 255), -1)
        for idx in self.landmark_indices["left_iris"]:
            landmark = self.landmarks[idx]
            image = cv2.circle(image, (int(landmark.x * width), int(landmark.y * height)), 2, (255, 255, 0), -1)
        for idx in self.landmark_indices["right_eyebrow"]:
            landmark = self.landmarks[idx]
            image = cv2.circle(image, (int(landmark.x * width), int(landmark.y * height)), 2, (255, 0, 0), -1)
        for idx in self.landmark_indices["left_eyebrow"]:
            landmark = self.landmarks[idx]
            image = cv2.circle(image, (int(landmark.x * width), int(landmark.y * height)), 2, (255, 0, 255), -1)
        image = cv2.flip(image, 1)
        cv2.imshow("image", image)

    def _compute_normal(self) -> Coordinates:
        """Computes the normal of the face"""
        faces = mediapipe_faces()
        normal = Coordinates(0, 0, 0)
        for i1, i2, i3 in faces:
            v1, v2, v3 = self.landmarks[i1], self.landmarks[i2], self.landmarks[i3]
            face_normal = Coordinates.cross_product(
                Coordinates(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z),
                Coordinates(v3.x - v1.x, v3.y - v1.y, v3.z - v1.z),
            ).normalize()
            if face_normal.z < 0:
                face_normal = face_normal * -1
            normal += face_normal
        return normal.normalize()

    def _local_eye_coords(self):
        # Get landmarks coords
        right_eye_coords = {
            idx: Coordinates(self.landmarks[idx].x, self.landmarks[idx].y) for idx in self.landmark_indices["right_eye"]
        }
        left_eye_coords = {
            idx: Coordinates(self.landmarks[idx].x, self.landmarks[idx].y) for idx in self.landmark_indices["left_eye"]
        }
        right_iris_coords = {
            idx: Coordinates(self.landmarks[idx].x, self.landmarks[idx].y)
            for idx in self.landmark_indices["right_iris"]
        }
        left_iris_coords = {
            idx: Coordinates(self.landmarks[idx].x, self.landmarks[idx].y) for idx in self.landmark_indices["left_iris"]
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

    def compute_transform(self) -> Mapping[str, Coordinates]:
        """Computes the position and rotation of the head with the coordinates of the landmarks"""
        position = Coordinates(
            sum([landmark_coords.x for landmark_coords in self.landmarks]) / len(self.landmarks),
            sum([landmark_coords.y for landmark_coords in self.landmarks]) / len(self.landmarks),
            sum([landmark_coords.z for landmark_coords in self.landmarks]) / len(self.landmarks),
        )
        eyes_coords = self._local_eye_coords()
        eyes_centers = {
            "right_eye": sum(eyes_coords["right_eye"].values(), Coordinates()) / len(eyes_coords["right_eye"]),
            "left_eye": sum(eyes_coords["left_eye"].values(), Coordinates()) / len(eyes_coords["left_eye"]),
            "right_iris": sum(eyes_coords["right_iris"].values(), Coordinates()) / len(eyes_coords["right_iris"]),
            "left_iris": sum(eyes_coords["left_iris"].values(), Coordinates()) / len(eyes_coords["left_iris"]),
        }

        return {
            "position": position,
            "rotation": self._compute_normal(),
            "eyes": eyes_coords,
            "eyes_centers": eyes_centers,
        }

    def calibrate(self) -> None:
        # numpy first 2 dimensions are switched
        instructions = {
            "center": "Look at the center of the screen",
            "move_left": "Move your head to the left side of the screen",
            "move_right": "Move your head to the right side of the screen",
            "look_left": "Look at the left side of the screen",
            "look_right": "Look at the right side of the screen",
            "look_up": "Look at the top of the screen",
            "look_down": "Look at the bottom of the screen",
        }
        for key, value in instructions.items():
            image = np.zeros((self.camera_dimensions.y, self.camera_dimensions.x, 3), np.uint8)
            image = cv2.putText(
                image,
                value,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("image", image)
            cv2.waitKey(0)
            while not self.get_frame_landmarks():
                cv2.waitKey(0)
            self.reference[key] = self.compute_transform()
        for key, value in self.reference.items():
            # print(key)
            # print(value["position"] - self.reference["center"]["position"], end="")
            # print(value["rotation"] - self.reference["center"]["rotation"])
            pass

    @property
    def looking_at(self) -> Direction:
        if self.landmarks is None:
            return Direction.CENTER
        return Direction.CENTER

    def run(self) -> None:
        while self.videoCapture.isOpened():
            self.get_frame_landmarks()
            self.draw_landmarks()
            self.looking_at
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.videoCapture.release()


if __name__ == "__main__":
    eye_detector = EyeDetector()
    eye_detector.run()
