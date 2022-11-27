import cv2
import mediapipe as mp

# mediapipe modules
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


class LandmarkDetector:
    def __init__(self) -> None:
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.videoCapture = cv2.VideoCapture(0)
        self.image = None
        self.landmarks = None

    def get_frame(self) -> bool:
        success, self.image = self.videoCapture.read()
        return success

    def draw_landmarks(self):
        # Draw face mesh connections
        mp_drawing.draw_landmarks(
            image=self.image,
            landmark_list=self.landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        # Draw eyebrows and eyelids
        mp_drawing.draw_landmarks(
            image=self.image,
            landmark_list=self.landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
        # Draw irises
        mp_drawing.draw_landmarks(
            image=self.image,
            landmark_list=self.landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    def run(self):
        while self.videoCapture.isOpened():
            if not self.get_frame():
                print("Ignoring empty camera frame.")
                continue

            self.image.flags.writeable = False
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(self.image)

            self.image.flags.writeable = True
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            # check if a face was detected
            if results.multi_face_landmarks:
                # draw the landmarks for the first face detected
                self.landmarks = results.multi_face_landmarks[0]
                self.draw_landmarks()
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow("MediaPipe Face Mesh", cv2.flip(self.image, 1))
            # Press escape to exit
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.videoCapture.release()


if __name__ == "__main__":
    landmark_detector = LandmarkDetector()
    landmark_detector.run()
