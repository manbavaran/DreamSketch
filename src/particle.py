# particle.py

import cv2
import numpy as np
import random

class Particle:
    def __init__(self, x, y, kind="star"):
        self.x = x
        self.y = y
        angle = random.uniform(-0.6, 0.6)
        speed = random.uniform(7, 15)
        self.vx = speed * np.cos(angle)
        self.vy = speed * np.sin(angle)
        self.size = random.randint(9, 15)
        self.life = 1.0
        self.kind = kind
        self.color = self.choose_color(kind)
        self.alpha = 1.0

    def choose_color(self, kind):
        if kind == "heart":
            return (random.randint(200,255), random.randint(70,120), random.randint(160,240))
        elif kind == "rose":
            return (random.randint(210,255), random.randint(70,120), random.randint(100,140))
        elif kind == "sakura":
            return (random.randint(240,255), random.randint(160,210), random.randint(220,255))
        else:
            return (random.randint(120, 255), random.randint(120,255), random.randint(180,255))

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.17
        self.alpha *= 0.97
        self.life -= 0.015

    def is_alive(self):
        return self.life > 0.05 and self.alpha > 0.09

    def draw(self, img):
        s = int(self.size * self.alpha)
        color = tuple(int(c * self.alpha + 255 * (1 - self.alpha)) for c in self.color)
        if self.kind == "heart":
            pts = self.heart_shape(self.x, self.y, s)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
            cv2.fillPoly(img, [pts], color)
        elif self.kind in ("rose", "sakura"):
            cv2.circle(img, (int(self.x), int(self.y)), s, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, (int(self.x), int(self.y)), s, color, -1, cv2.LINE_AA)

    def heart_shape(self, x, y, s):
        # 하트 shape (간단 2D path)
        t = np.linspace(0, 2*np.pi, 100)
        pts = np.array([
            (
                x + s*16*np.sin(tt)**3,
                y - s*(13*np.cos(tt) - 5*np.cos(2*tt) - 2*np.cos(3*tt) - np.cos(4*tt))
            ) for tt in t
        ], dtype=np.int32)
        return pts

class ParticleSystem:
    def __init__(self):
        self.particles = []

    def emit(self, x, y, n=20, kind="star"):
        for _ in range(n):
            self.particles.append(Particle(x, y, kind=kind))

    def update_and_draw(self, frame):
        overlay = frame.copy()
        next_particles = []
        for p in self.particles:
            p.update()
            if p.is_alive():
                p.draw(overlay)
                next_particles.append(p)
        self.particles = next_particles
        out = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        return out
