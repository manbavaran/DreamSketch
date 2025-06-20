import cv2
import numpy as np
from collections import deque

# 화려한 팔레트로 색상 그라데이션
palette = [
    (255, 120, 220),  # 핑크
    (255, 200, 100),  # 노랑
    (100, 180, 255),  # 하늘
    (180, 120, 255),  # 보라
    (255, 100, 160),  # 연핑크
]

def get_grad_color(idx, total):
    ratio = idx / max(1, total - 1)
    color_idx = int(ratio * (len(palette) - 1))
    color1 = np.array(palette[color_idx])
    color2 = np.array(palette[min(color_idx + 1, len(palette) - 1)])
    t = (ratio * (len(palette) - 1)) % 1.0
    col = color1 * (1 - t) + color2 * t
    return tuple(int(x) for x in col)

class TrajectoryGlow:
    def __init__(self, max_trail=120):
        self.trail = deque(maxlen=max_trail)
        self.active = False

    def add_point(self, point):
        self.trail.append(point)

    def clear(self):
        self.trail.clear()

    def draw(self, frame, brighten=0.0, fade_out=False):
        overlay = frame.copy()
        n = len(self.trail)
        if n == 0: return frame
        # 궤적 그라데이션, 선 두께/알파 변화
        for i in range(1, n):
            x1, y1 = self.trail[i-1]
            x2, y2 = self.trail[i]
            color = get_grad_color(i, n)
            thickness = int(13 * (1 - i / n) + 6)
            alpha = 0.7 + brighten if not fade_out else max(0.18, 0.85*(n-i)/n)
            # 밝은 Glow 효과: 선 위에 투명하게 겹치기
            cv2.line(overlay, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        # 마지막 포인트 glow: 원 3겹 (빛나는 효과)
        x, y = self.trail[-1]
        cv2.circle(overlay, (x, y), 19, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 13, (255, 210, 240), -1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 8, (255, 255, 180), -1, cv2.LINE_AA)
        # 전체적으로 Blur 느낌 추가 (약간의 흐림)
        out = cv2.addWeighted(overlay, 0.75 + brighten, frame, 0.25 - brighten, 0)
        return out
