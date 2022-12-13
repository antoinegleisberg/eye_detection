import pygame
from pygame.locals import QUIT
import sys
from eye_detector import EyeDetector
import cv2
import os
import numpy as np
import random


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


class Game:
    def __init__(self):
        os.environ["SDL_VIDEO_WINDOW_POS"] = "0,30"
        self.eye_input = EyeDetector()
        pygame.init()
        self.display = pygame.display.set_mode((1920, 1000))
        self.clock = pygame.time.Clock()
        self.player = Player(960, 500, 50, 50, (0, 0, 255))

        self.background = np.empty((3840, 2000, 3))
        self._init_background()

        self.video_capture = cv2.VideoCapture(0)
        self.videoWriter = cv2.VideoWriter(
            "minigame_video.avi",
            cv2.VideoWriter_fourcc(*"MJPG"),
            10,
            (int(self.video_capture.get(3)), int(self.video_capture.get(4))),
        )

    def init_ennemies(self, number):
        ennemies = []
        for _ in range(number):
            x = random.randint(0, 1920)
            y = random.randint(0, 1000)
            ennemies.append(Ennemy(960 + x, 500 + y, 50, 50, (255, 0, 0)))
        return ennemies

    def _init_background(self):
        # Set the background color to brown
        self.background[:, :, 0] = 150
        self.background[:, :, 1] = 75
        self.background[:, :, 2] = 0
        # Draw the grass
        self.background[960:2880, 500:1500, 0] = 24
        self.background[960:2880, 500:1500, 1] = 164
        self.background[960:2880, 500:1500, 2] = 86
        # Draw random darker spots on the grass. They are 20x20 pixels
        for i in range(1920 // 20):
            for j in range(1000 // 20):
                if random.random() < 0.1:
                    x = 960 + 20 * i
                    y = 500 + 20 * j
                    self.background[x : x + 20, y : y + 20, 0] = 24 * 0.8  # noqa
                    self.background[x : x + 20, y : y + 20, 1] = 164 * 0.8  # noqa
                    self.background[x : x + 20, y : y + 20, 2] = 86 * 0.8  # noqa

    def run(self):
        bullets = []
        ennemies = self.init_ennemies(5)
        while True:
            # check for events
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

            # update game
            # get input
            success, image = self.video_capture.read()
            if not success:
                continue

            self.eye_input.set_image(image)
            self.videoWriter.write(image)
            blinking = self.eye_input.blinked
            direction = self.eye_input.looking_at

            # update player position
            player_x_position, player_y_position = self.player.get_position()
            new_player_x_position = clamp(player_x_position + self.player.speed * direction.value[0], 25, 1895)
            new_player_y_position = clamp(player_y_position + self.player.speed * direction.value[1], 25, 975)
            self.player.set_position(new_player_x_position, new_player_y_position)

            # if blinking : shoot bullet
            if blinking:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                bullets.append(
                    Bullet(
                        new_player_x_position + 960,
                        new_player_y_position + 500,
                        new_player_x_position,
                        new_player_y_position,
                        mouse_x,
                        mouse_y,
                        (0, 0, 0),
                    )
                )
            for bullet in bullets:
                bullet.move()
            bullets, ennemies = detect_collision(bullets, ennemies)

            # render game
            self.display.blit(
                pygame.surfarray.make_surface(self.background), (-new_player_x_position, -new_player_y_position)
            )
            self.player.draw(self.display)
            for bullet in bullets:
                bullet.draw(self.display, (-new_player_x_position + 960, -new_player_y_position + 500))
            for ennemy in ennemies:
                ennemy.draw(self.display, (-new_player_x_position, -new_player_y_position))

            self.clock.tick(60)
            pygame.display.update()


class Player:
    def __init__(self, x, y, width, height, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.speed = 20

    def get_position(self):
        return self.x, self.y

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def draw(self, display):
        pygame.draw.rect(display, self.color, (960 - self.width / 2, 500 - self.height / 2, self.width, self.height))


class Ennemy(Player):
    def draw(self, display, offset):
        pygame.draw.rect(display, self.color, (self.x + offset[0], self.y + offset[1], self.width, self.height))


class Bullet:
    def __init__(self, x, y, screen_x, screen_y, mouse_x, mouse_y, color):
        self.x = x  # position in the world
        self.y = y
        self.screen_x = screen_x  # position on the screen
        self.screen_y = screen_y
        self.radius = 10
        self.angle = np.arctan2(mouse_y - 500, mouse_x - 960)
        self.color = color
        self.speed = 20

    def move(self):
        self.screen_x += self.speed * np.cos(self.angle)
        self.screen_y += self.speed * np.sin(self.angle)
        self.x += self.speed * np.cos(self.angle)
        self.y += self.speed * np.sin(self.angle)
        print(self.x, self.y)

    def get_position(self):
        return self.x, self.y

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def draw(self, display, offset):
        pygame.draw.circle(display, self.color, (self.screen_x + offset[0], self.screen_y + offset[1]), self.radius, 0)


def detect_collision(bullets, ennemies):
    n_bullet = 0
    while n_bullet < len(bullets):
        bullet = bullets[n_bullet]
        for ennemy in ennemies:
            ennemy_min_x, ennemy_min_y = ennemy.get_position()
            ennemy_max_x, ennemy_max_y = ennemy_min_x + ennemy.width, ennemy_min_y + ennemy.height
            if (
                bullet.x > ennemy_min_x
                and bullet.x < ennemy_max_x
                and bullet.y > ennemy_min_y
                and bullet.y < ennemy_max_y
            ):
                print("collision")
                ennemies.remove(ennemy)
                bullets.remove(bullet)
                break
        else:
            n_bullet += 1
    return bullets, ennemies


if __name__ == "__main__":
    game = Game()
    game.run()
