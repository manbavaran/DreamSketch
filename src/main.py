# main.py

import cv2
import mediapipe as mp
from gesture import is_ok_sign, is_index_finger_up, is_fist, is_hand_open, is_heart_gesture
from trajectory_glow import TrajectoryGlow
from particle import ParticleSystem

def main():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    traj = TrajectoryGlow()
    particles = ParticleSystem()
    mode = "idle"
    idle_timer, finish_timer = 0, 0
    last_open = {"left":False, "right":False}
    rose_triggered = sakura_triggered = False

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        hands_lms = results.multi_hand_landmarks

        # 제스처 인식 (mode 전이)
        if mode == "idle":
            if is_ok_sign(hands_lms):
                mode = "ready"
        elif mode == "ready":
            if is_index_finger_up(hands_lms):
                mode = "draw"
                traj.active = True
            elif is_ok_sign(hands_lms):
                mode = "idle"
        elif mode == "draw":
            if is_ok_sign(hands_lms):
                mode = "finish"
                finish_timer = cv2.getTickCount()
                traj.active = False

        # 글씨쓰기 모드
        if mode == "draw" and hands_lms:
            lm = hands_lms[0]  # 한 손만
            ix = int(lm.landmark[8].x * w)
            iy = int(lm.landmark[8].y * h)
            traj.add_point((ix, iy))

        # 효과/이펙트 트리거 (파티클/하트/꽃잎)
        # 손바닥 open (유성우: star), 하트 (heart), 왼손 rose, 오른손 sakura
        if hands_lms:
            # 하트(양손)
            if is_heart_gesture(hands_lms):
                particles.emit(w//2, h//2, n=10, kind="heart")
            # 주먹→펴짐(꽃잎)
            left, right = hands_lms[0], (hands_lms[1] if len(hands_lms)==2 else None)
            # 왼손 꽃(rose): 주먹후 펴면, 오른손 벚꽃(sakura): 주먹후 펴면
            for idx, (lm, name) in enumerate(zip([left,right],[ "left", "right"])):
                if lm:
                    if is_fist([lm]):
                        last_open[name] = False
                    elif is_hand_open([lm]) and not last_open[name]:
                        if name=="left":
                            particles.emit(int(0.25*w), int(0.7*h), n=14, kind="rose")
                        elif name=="right":
                            particles.emit(int(0.75*w), int(0.7*h), n=14, kind="sakura")
                        last_open[name] = True

            # 손바닥 open (유성우)
            for lm in hands_lms:
                if is_hand_open([lm]):
                    cx = int(lm.landmark[9].x * w)
                    cy = int(lm.landmark[9].y * h)
                    particles.emit(cx, cy, n=2, kind="star")

        # 효과 적용
        if mode == "finish":
            elapsed = (cv2.getTickCount() - finish_timer)/cv2.getTickFrequency()
            out = traj.draw(frame, brighten=min(1.0, elapsed/1.5))
            if elapsed > 1.5:
                out = traj.draw(out, fade_out=True)
                if elapsed > 3.5:
                    traj.clear()
                    mode = "idle"
        elif mode == "draw":
            out = traj.draw(frame)
        else:
            out = frame.copy()

        out = particles.update_and_draw(out)

        # UI
        cv2.putText(out, f"Mode: {mode}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 180, 255), 2, cv2.LINE_AA)
        cv2.imshow("DreamSketch", out)
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
