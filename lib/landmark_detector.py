import cv2
import mediapipe as mp

# mediapipe modules
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


class LandmarkDetector:
    def __init__(self, dynamic=False, to_video=False) -> None:
        """Initialize the landmark detector. If dynamic is True, the detector will use the webcam"""
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.image = None
        self.landmarks = None
        self.dynamic = dynamic
        self.to_video = to_video
        if self.dynamic:
            self.videoCapture = cv2.VideoCapture(0)
            self.videoWriter = cv2.VideoWriter(
                "filename.avi",
                cv2.VideoWriter_fourcc(*"MJPG"),
                10,
                (int(self.videoCapture.get(3)), int(self.videoCapture.get(4))),
            )

    def set_image(self, image):
        """Set the image to be processed"""
        self.image = image

    def get_frame(self) -> bool:
        """Get a frame from the video capture; return True if successful"""
        assert self.dynamic, "Cannot get a frame for a static setting"
        success, self.image = self.videoCapture.read()
        return success

    def process_image(self):
        """Get the landmarks from the image and return the results"""
        self.image.flags.writeable = False
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(self.image)
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        return results

    def draw_landmarks(self):
        """Draws the landmarks on the image"""
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

    def show_landmarks(self):
        """Show the image with the landmarks"""
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow("MediaPipe Face Mesh", cv2.flip(self.image, 1))

    def run(self):
        assert self.dynamic, "Cannot run a static image"
        while self.videoCapture.isOpened():
            if not self.get_frame():
                print("Ignoring empty camera frame.")
                continue

            results = self.process_image()
            if results.multi_face_landmarks != []:  # check if a face was detected
                self.landmarks = results.multi_face_landmarks[0]
                self.draw_landmarks()  # draw the landmarks for the first face detected
                self.show_landmarks()

            if self.to_video:
                self.videoWriter.write(self.image)

            # Press escape to exit
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.videoCapture.release()
        if self.to_video:
            self.videoWriter.release()


if __name__ == "__main__":
    landmark_detector = LandmarkDetector(dynamic=True, to_video=True)
    landmark_detector.run()
