# particle.py
import cv2
import numpy as np
import random

class Particle:
    def __init__(self, x, y, angle, speed, color, size=8, fade=1.0):
        self.x = x
        self.y = y
        self.vx = speed * np.cos(angle)
        self.vy = speed * np.sin(angle)
        self.color = color
        self.size = size
        self.alpha = fade  # 1.0~0.0
        self.life = 1.0    # 1.0~0.0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.18  # gravity
        self.alpha *= 0.97
        self.life -= 0.017

    def is_alive(self):
        return self.life > 0.0 and self.alpha > 0.1

def random_color():
    base = np.array([random.randint(120, 255) for _ in range(3)])
    return tuple(map(int, base))

def run_particle_demo():
    cap = cv2.VideoCapture(0)
    particles = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        key = cv2.waitKey(1)
        if key == ord(' '):
            for _ in range(20):
                angle = random.uniform(-0.5, 0.5)
                speed = random.uniform(8, 15)
                color = random_color()
                particles.append(Particle(w//2, h//2, angle, speed, color, size=random.randint(8,13)))

        overlay = frame.copy()
        new_particles = []
        for p in particles:
            p.update()
            if p.is_alive():
                alpha = min(1.0, max(0.0, p.alpha))
                s = int(p.size * alpha)
                color = [int(c * alpha + 255 * (1 - alpha)) for c in p.color]
                cv2.circle(overlay, (int(p.x), int(p.y)), s, color, -1, cv2.LINE_AA)
                new_particles.append(p)
        particles = new_particles

        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        cv2.imshow("Meteor Particle Effect", frame)

        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
