import cv2
import mediapipe as mp
import time
from gesture import (
    is_ok_sign, is_index_finger_up, is_heart_gesture,
    is_vertical_sweep_gesture, is_fist_palm_flip_seq
)
from trajectory_glow import TrajectoryGlow
from particle import ParticleSystem

def main():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.6, min_tracking_confidence=0.6
    )
    traj = TrajectoryGlow(max_trail=120)
    particles = ParticleSystem()
    mode = "idle"
    fade_stage = None
    fade_t0 = None

    # 쿨타임용
    last_heart = 0
    last_meteor = 0
    last_rose = 0
    prev_sweep_x = {"left": None, "right": None}
    prev_flip_state = {"left": None, "right": None}
    ok_since = None
    OK_HOLD_SEC = 0.25

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        hands_lms = results.multi_hand_landmarks
        t_now = time.time()

        # ---------- 상태 전이 (글씨쓰기용) ----------
        if mode == "idle":
            if is_ok_sign(hands_lms):
                if ok_since is None:
                    ok_since = t_now
                elif t_now - ok_since > OK_HOLD_SEC:
                    mode = "ready"
                    ok_since = None
            else:
                ok_since = None
        elif mode == "ready":
            if is_index_finger_up(hands_lms):
                mode = "draw"
                traj.active = True
            elif is_ok_sign(hands_lms):
                mode = "idle"
        elif mode == "draw":
            if is_ok_sign(hands_lms):
                mode = "finish"
                fade_stage = "brighten"
                fade_t0 = time.time()
                traj.active = False

        # ---------- 글씨쓰기 (draw) ----------
        if mode == "draw" and hands_lms:
            lm = hands_lms[0]
            ix = int(lm.landmark[8].x * w)
            iy = int(lm.landmark[8].y * h)
            traj.add_point((ix, iy))
            particles.emit(ix, iy, n=1, kind="star")

        # ---------- idle 모드에서만 이펙트 ----------
        if mode == "idle" and hands_lms:
            # --- 하트 ---
            if is_heart_gesture(hands_lms) and (t_now - last_heart > 2):
                particles.emit(w//2, h//2, n=30, kind="heart")
                last_heart = t_now

            # --- 유성우(손날 스윕) ---
            for idx, lm in enumerate(hands_lms):
                hand_label = "left" if idx == 0 else "right"
                cx = int(lm.landmark[9].x * w)
                detected = is_vertical_sweep_gesture(lm, prev_sweep_x[hand_label], w)
                if detected and (t_now - last_meteor > 1.0):
                    # 스윕 방향으로 유성우 각도 표현 가능 (추후 개선)
                    particles.emit(cx, int(0.4*h), n=55, kind="star")
                    last_meteor = t_now
                prev_sweep_x[hand_label] = cx

            # --- 꽃잎(주먹→손바닥 뒤집기) ---
            for idx, lm in enumerate(hands_lms):
                hand_label = "left" if idx == 0 else "right"
                triggered, new_state = is_fist_palm_flip_seq(lm, prev_flip_state[hand_label])
                if triggered and (t_now - last_rose > 2):
                    if hand_label == "left":
                        particles.emit(int(0.25*w), int(0.7*h), n=24, kind="rose")
                    elif hand_label == "right":
                        particles.emit(int(0.75*w), int(0.7*h), n=24, kind="sakura")
                    last_rose = t_now
                prev_flip_state[hand_label] = new_state

        # ---------- 글씨/효과 적용 및 fade ----------
        out = frame.copy()
        if mode == "draw":
            out = traj.draw(out, brighten=0.0, fade_out=False)
        elif mode == "finish":
            t = time.time() - fade_t0
            if fade_stage == "brighten":
                alpha = min(1.0, t / 1.4)
                out = traj.draw(out, brighten=alpha, fade_out=False)
                if alpha >= 1.0:
                    fade_stage = "hold"
                    fade_t0 = time.time()
            elif fade_stage == "hold":
                out = traj.draw(out, brighten=1.0, fade_out=False)
                if (time.time() - fade_t0) > 1.2:
                    fade_stage = "fadeout"
                    fade_t0 = time.time()
            elif fade_stage == "fadeout":
                alpha = max(0, 1.0 - (time.time() - fade_t0) / 2.0)
                out = traj.draw(out, brighten=alpha, fade_out=True)
                if alpha <= 0:
                    traj.clear()
                    mode = "idle"
                    fade_stage = None
                    fade_t0 = None

        out = particles.update_and_draw(out)
        cv2.putText(out, f"Mode: {mode}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 180, 255), 2, cv2.LINE_AA)
        cv2.imshow("DreamSketch", out)
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
