# trajectory_glow.py
import cv2
import numpy as np
import mediapipe as mp
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

def run_trajectory_demo():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    MAX_TRAIL = 50
    trail = deque(maxlen=MAX_TRAIL)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            ix = int(lm.landmark[8].x * w)
            iy = int(lm.landmark[8].y * h)
            trail.append((ix, iy))

        overlay = frame.copy()
        for i in range(1, len(trail)):
            x1, y1 = trail[i-1]
            x2, y2 = trail[i]
            color = get_grad_color(i, len(trail))
            thickness = int(14 * (1 - i / len(trail)) + 6)
            cv2.line(
                overlay, (x1, y1), (x2, y2),
                color, thickness, cv2.LINE_AA
            )

        if len(trail) > 0:
            x, y = trail[-1]
            cv2.circle(overlay, (x, y), 18, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, (x, y), 13, (255, 210, 240), -1, cv2.LINE_AA)
            cv2.circle(overlay, (x, y), 7, (255, 255, 160), -1, cv2.LINE_AA)

        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cv2.imshow("Glow Trajectory", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
