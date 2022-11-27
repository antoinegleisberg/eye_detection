import cv2
from screeninfo import get_monitors
import numpy as np
import random
from pathlib import Path
import csv


class ImageGenerator:
    def __init__(self, circle_size: int = 10) -> None:
        self.screen_width = get_monitors()[0].width
        self.screen_height = get_monitors()[0].height
        self.image = np.full((self.screen_height, self.screen_width, 3), 255, np.uint8)
        self.videoCapture = self.launch()
        self.point_size = circle_size
        self.csvfile = None
        self.csv_reader = None
        self.csv_writer = None
        self.next_id = 0
        self.data_folder = Path("data")
        self.image_folder = Path("data/images")
        self.dataset_name = Path("dataset.csv")

    def init_csv(self):
        self.csvfile = open(self.data_folder / self.dataset_name, "a+")
        self.csv_reader = csv.reader(self.csvfile)
        self.csv_writer = csv.writer(self.csvfile)
        self.next_id = len(list(self.image_folder.iterdir()))

    def launch(self):
        cv2.namedWindow("Dataset Generator")
        videoCapture = cv2.VideoCapture(0)
        if not videoCapture.isOpened():
            raise Exception("Starting video capture failed")
        return videoCapture

    def save(self, x: int, y: int):
        ret, frame = self.videoCapture.read()
        if not ret:
            return
        self.csv_writer.writerow([f"{self.next_id}.png", x, y])
        cv2.imwrite(str(self.image_folder / Path(f"{self.next_id}.png")), frame)
        self.next_id += 1

    def show_random_point(self):
        x, y = (
            random.randint(self.point_size, self.screen_width - self.point_size),
            random.randint(self.point_size, self.screen_height - self.point_size),
        )
        self.image = np.full((self.screen_height, self.screen_width, 3), 255, np.uint8)
        self.image = cv2.circle(self.image, (x, y), self.point_size, (255, 0, 0), -1)
        cv2.imshow("Dataset Generator", self.image)
        return x, y

    def show_instructions(self):
        self.image = cv2.putText(
            self.image,
            "Press Enter to save the image ; Press Esc to exit ; Press any key to continue",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Dataset Generator", self.image)

    def run(self):
        self.init_csv()
        self.show_instructions()
        cv2.waitKey(0)
        x, y = self.show_random_point()
        while True:
            key = cv2.waitKey(20)
            if key == 27:
                break
            elif key == 13:
                self.save(x, y)
                x, y = self.show_random_point()

        self.csvfile.close()
        self.videoCapture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    generator = ImageGenerator()
    generator.run()
