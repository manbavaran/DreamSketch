# trajectory_glow.py

import cv2
import numpy as np
from collections import deque

palette = [
    (255, 120, 220),
    (255, 200, 100),
    (100, 180, 255),
    (180, 120, 255)
]

def get_grad_color(idx, total):
    ratio = idx / max(1, total - 1)
    color_idx = int(ratio * (len(palette) - 1))
    color1 = np.array(palette[color_idx])
    color2 = np.array(palette[min(color_idx + 1, len(palette) - 1)])
    t = (ratio * (len(palette) - 1)) % 1.0
    return tuple(np.uint8(color1 * (1 - t) + color2 * t))

class TrajectoryGlow:
    def __init__(self, max_trail=50):
        self.trail = deque(maxlen=max_trail)
        self.active = False  # 글씨쓰기 모드

    def add_point(self, point):
        self.trail.append(point)

    def clear(self):
        self.trail.clear()

    def draw(self, frame, brighten=0.0, fade_out=False):
        overlay = frame.copy()
        n = len(self.trail)
        if n == 0: return frame
        # 그라데이션 궤적 그리기
        for i in range(1, n):
            x1, y1 = self.trail[i-1]
            x2, y2 = self.trail[i]
            color = get_grad_color(i, n)
            thickness = int(14 * (1 - i / n) + 6)
            alpha = 0.6 + brighten if not fade_out else max(0.2, 0.8*(n-i)/n)
            cv2.line(overlay, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        # 마지막 glow
        x, y = self.trail[-1]
        cv2.circle(overlay, (x, y), 18, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 13, (255, 210, 240), -1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 7, (255, 255, 160), -1, cv2.LINE_AA)
        out = cv2.addWeighted(overlay, 0.7 + brighten, frame, 0.3 - brighten, 0)
        return out
