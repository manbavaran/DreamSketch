import cv2
import numpy as np
import random
import math

def draw_star(img, x, y, size, color, angle=0, thickness=2):
    pts = []
    for i in range(5):
        theta = angle + i * 2 * np.pi / 5
        r = size
        sx = x + r * np.sin(theta)
        sy = y - r * np.cos(theta)
        pts.append((int(sx), int(sy)))
    for i in range(5):
        cv2.line(img, pts[i], pts[(i+2)%5], color, thickness, cv2.LINE_AA)

class MeteorParticle:
    def __init__(self, x, y, vx, vy, size, color):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.size = size
        self.color = color
        self.alpha = 1.0
        self.trail = []
        self.life = 1.0

    def update(self):
        self.trail.append((self.x, self.y, self.alpha))
        if len(self.trail) > 20:
            self.trail.pop(0)
        self.x += self.vx
        self.y += self.vy
        self.alpha *= 0.94
        self.life -= 0.014

    def is_alive(self):
        return self.life > 0.04 and self.alpha > 0.09

    def draw(self, img):
        for i in range(1, len(self.trail)):
            x1, y1, a1 = self.trail[i-1]
            x2, y2, a2 = self.trail[i]
            c = tuple(int(a1 * c + (1-a1)*255) for c in self.color)
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 4, cv2.LINE_AA)
        draw_star(img, int(self.x), int(self.y), int(self.size), self.color, angle=random.uniform(0, 2*math.pi), thickness=2)

class Particle:
    def __init__(self, x, y, kind="star", vx=None, vy=None):
        self.x = x
        self.y = y
        self.kind = kind
        angle = random.uniform(-0.9, 0.9)
        speed = random.uniform(8, 18)
        self.vx = vx if vx is not None else speed * np.cos(angle)
        self.vy = vy if vy is not None else speed * np.sin(angle)
        self.size = random.randint(10, 17)
        self.life = 1.0
        self.color = self.choose_color(kind)
        self.alpha = 1.0
        self.grow = 0.1 if kind == "heart" else 0

    def choose_color(self, kind):
        if kind == "heart":
            return (random.randint(210,255), random.randint(70,120), random.randint(160,240))
        elif kind == "rose":
            return (random.randint(230,255), random.randint(70,120), random.randint(100,150))
        elif kind == "sakura":
            return (random.randint(240,255), random.randint(170,220), random.randint(230,255))
        else:
            colors = [
                (255, 255, 180), (200, 160, 255),
                (120, 220, 255), (255, 180, 210),
                (255, 240, 120)
            ]
            return random.choice(colors)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.17
        self.alpha *= 0.97
        self.life -= 0.012
        self.size += self.grow

    def is_alive(self):
        return self.life > 0.06 and self.alpha > 0.10

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
            draw_star(img, int(self.x), int(self.y), s, color, angle=random.uniform(0,2*np.pi), thickness=2)

    def heart_shape(self, x, y, s):
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

    def emit(self, x, y, n=22, kind="star"):
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
        out = cv2.addWeighted(overlay, 0.74, frame, 0.26, 0)
        return out
    
    def emit_meteor_grid(self, w, h, direction="right", n_col=7, n_row=3):
        for col in range(n_col):
            for row in range(n_row):
                # 별똥별 시작 위치를 더 퍼지게 (±0.08 랜덤)
                base_x = col / (n_col - 1)
                base_y = row / (n_row + 1)
                if direction == "right":
                    start_x = int(w * (base_x + random.uniform(-0.08, 0.08)))
                    start_y = int(h * (base_y + random.uniform(-0.06, 0.06)))
                    angle = np.radians(67 + random.uniform(-3, 3))
                else:
                    start_x = int(w * (1 - base_x + random.uniform(-0.08, 0.08)))
                    start_y = int(h * (base_y + random.uniform(-0.06, 0.06)))
                    angle = np.radians(113 + random.uniform(-3, 3))
                speed = random.uniform(20, 25)
                vx = speed * np.cos(angle)
                vy = speed * np.sin(angle)
                size = random.randint(12, 19)
                color = (255,255,255)
                self.particles.append(MeteorParticle(start_x, start_y, vx, vy, size, color))
                
    def emit_flower_burst(self, x, y, kind="rose", n=40, spread=1.2):
        for i in range(n):
            angle = 2 * np.pi * i / n + random.uniform(-0.1, 0.1)
            speed = random.uniform(14, 23) * spread
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            self.particles.append(Particle(x, y, kind=kind, vx=vx, vy=vy))

