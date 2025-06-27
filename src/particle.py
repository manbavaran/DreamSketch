import numpy as np
import random
import cv2
import time

def draw_star(img, center, radius, color):
    pts = []
    angle_offset = -np.pi/2
    for i in range(5):
        angle = angle_offset + i * 2 * np.pi / 5
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        pts.append((x, y))
        angle = angle_offset + (i + 0.5) * 2 * np.pi / 5
        x = int(center[0] + (radius * 0.43) * np.cos(angle))
        y = int(center[1] + (radius * 0.43) * np.sin(angle))
        pts.append((x, y))
    pts = np.array(pts, np.int32)
    cv2.fillPoly(img, [pts], color)

class Particle:
    def __init__(self, x, y, vx, vy, length, t0, kind="meteor", start_delay=0):
        self.x = float(x)
        self.y = float(y)
        self.vx = vx
        self.vy = vy
        self.length = length
        self.kind = kind
        self.t0 = t0
        self.start_delay = start_delay
        self.started = False
        self.life = 1.0
        self.alpha = 1.0
        self.angle = random.uniform(0, 2*np.pi)
        self.rot_speed = random.uniform(-0.13, 0.13)
        self.radius = 14 if kind == "meteor" else random.randint(8, 16)

    def update(self):
        now = time.time()
        if not self.started and now - self.t0 < self.start_delay:
            return
        self.started = True
        self.x += self.vx
        self.y += self.vy
        if self.kind == "meteor":
            self.life -= 0.013
        else:
            self.life -= 0.019
        self.alpha = max(0, self.life)
        self.angle += self.rot_speed
        if self.kind in ["rose", "sakura", "heart"]:
            self.vy += 0.15

    def is_alive(self):
        if not self.started:
            return True
        return self.life > 0 and (0 <= self.x < 4000) and (0 <= self.y < 4000)

    def draw(self, img):
        if not self.started:
            return
        if self.kind == "meteor":
            x1, y1 = int(self.x), int(self.y)
            angle = np.arctan2(self.vy, self.vx)
            length = int(self.length * self.alpha)
            x2 = int(x1 - length * np.cos(angle))
            y2 = int(y1 - length * np.sin(angle))
            steps = 14
            for i in range(steps):
                frac0 = i/steps
                frac1 = (i+1)/steps
                sx0 = int(x1*(1-frac0) + x2*frac0)
                sy0 = int(y1*(1-frac0) + y2*frac0)
                sx1 = int(x1*(1-frac1) + x2*frac1)
                sy1 = int(y1*(1-frac1) + y2*frac1)
                brightness = int(255 * (1-frac0)**1.5 * self.alpha)
                cv2.line(img, (sx0, sy0), (sx1, sy1), (255,255,255), thickness=2)
            # ★ 머리는 별
            draw_star(img, (x1, y1), 12, (255,255,255))
        elif self.kind == "heart":
            self._draw_heart(img, int(self.radius * self.alpha), (230, 60, 170))
        elif self.kind == "rose":
            self._draw_flower(img, int(self.radius * self.alpha), (80,0,200), petal=8)
        elif self.kind == "sakura":
            self._draw_flower(img, int(self.radius * self.alpha), (255,180,240), petal=6)

    def _draw_heart(self, img, r, color):
        center = (int(self.x), int(self.y))
        pts = []
        for t in np.linspace(0, 2*np.pi, 80):
            x = 16 * np.sin(t) ** 3
            y = 13 * np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)
            pts.append([center[0] + int(x * r/18), center[1] - int(y * r/18)])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(img, [pts], color)

    def _draw_flower(self, img, r, color, petal=8):
        center = (int(self.x), int(self.y))
        for i in range(petal):
            angle = self.angle + 2*np.pi*i/petal
            px = int(center[0] + np.cos(angle)*r)
            py = int(center[1] + np.sin(angle)*r)
            cv2.ellipse(img, (px, py), (r//3, r//2), angle*180/np.pi, 0, 360, color, -1)
        cv2.circle(img, center, max(2, r//3), color, -1)

class ParticleSystem:
    def __init__(self):
        self.particles = []

    def emit(self, x, y, n=1, kind="star"):
        for _ in range(n):
            angle = random.uniform(0, 2*np.pi)
            speed = random.uniform(8, 14)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            self.particles.append(Particle(x, y, vx, vy, 0, time.time(), kind=kind))

    def emit_flower_burst(self, x, y, kind="rose", n=40, spread=1.2):
        for i in range(n):
            angle = 2 * np.pi * i / n + random.uniform(-0.1, 0.1)
            speed = random.uniform(14, 23) * spread
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            self.particles.append(Particle(x, y, vx, vy, 0, time.time(), kind=kind))


    def emit_meteor_rain(self, w, h, direction="dr", n=22):
        y0 = int(h * 0.04)
        angle_deg = 66 if direction == "dr" or direction == "right" else 114
        angle = np.radians(angle_deg)
        for i in range(n):
            frac = i / (n - 1)
            x0 = int(w * frac) if direction in ("dr", "right") else int(w * (1 - frac))
            start_delay = random.uniform(0, 0.7) + 0.10 * frac
            speed = 25
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            self.particles.append(Particle(x0, y0, vx, vy, length=int(0.8*h), t0=time.time(), kind="meteor", start_delay=start_delay))


    def update_and_draw(self, frame):
        new_particles = []
        for p in self.particles:
            p.update()
            if p.is_alive():
                p.draw(frame)
                new_particles.append(p)
        self.particles = new_particles
        return frame
