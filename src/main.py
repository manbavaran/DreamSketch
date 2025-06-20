import cv2
import mediapipe as mp
import time
from gesture import (
    is_ok_sign, is_index_finger_up, is_heart_gesture,
    is_fist_palm_flip_seq
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

    last_heart = 0
    last_meteor = 0
    last_rose = 0
    prev_flip_state = {"left": None, "right": None}
    ok_since = None
    OK_HOLD_SEC = 0.25

    # 손날 스윕 누적
    sweep_state = {"left": None, "right": None}
    SWEEP_TRIGGER_TIME = 1.5
    SWEEP_MIN_DIST = 0.15

    cv2.namedWindow("DreamSketch", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("DreamSketch", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        hands_lms = results.multi_hand_landmarks
        t_now = time.time()

        # 글씨쓰기 상태 전이
        if mode == "idle":
            if hands_lms and len(hands_lms) == 1 and is_ok_sign(hands_lms):
                if ok_since is None:
                    ok_since = t_now
                elif t_now - ok_since > OK_HOLD_SEC:
                    mode = "ready"
                    ok_since = None
            else:
                ok_since = None
        elif mode == "ready":
            if hands_lms and len(hands_lms) == 1 and is_index_finger_up(hands_lms):
                mode = "draw"
                traj.active = True
            elif hands_lms and len(hands_lms) == 1 and is_ok_sign(hands_lms):
                mode = "idle"
        elif mode == "draw":
            if hands_lms and len(hands_lms) == 1 and is_ok_sign(hands_lms):
                mode = "finish"
                fade_stage = "brighten"
                fade_t0 = time.time()
                traj.active = False

        # 글씨쓰기
        if mode == "draw" and hands_lms and len(hands_lms) == 1:
            lm = hands_lms[0]
            ix = int(lm.landmark[8].x * w)
            iy = int(lm.landmark[8].y * h)
            traj.add_point((ix, iy))
            particles.emit(ix, iy, n=1, kind="star")

        # idle 모드 이펙트
        if mode == "idle" and hands_lms:
            # 하트(두 손)
            if len(hands_lms) == 2:
                if is_heart_gesture(hands_lms) and (t_now - last_heart > 2):
                    particles.emit(w//2, h//2, n=30, kind="heart")
                    last_heart = t_now

            # 손날 스윕(1.5초 이상)
            for idx, lm in enumerate(hands_lms):
                hand_label = "left" if idx == 0 else "right"
                cx = lm.landmark[9].x
                open_palm = all(lm.landmark[tid].y < lm.landmark[tid-2].y for tid in [8,12,16,20])
                if open_palm:
                    now = time.time()
                    if sweep_state[hand_label] is None:
                        sweep_state[hand_label] = {"start_x": cx, "start_time": now}
                    else:
                        dx = cx - sweep_state[hand_label]["start_x"]
                        dt = now - sweep_state[hand_label]["start_time"]
                        if dt > SWEEP_TRIGGER_TIME and abs(dx) > SWEEP_MIN_DIST:
                            direction = "right" if dx > 0 else "left"
                            if t_now - last_meteor > 1.5:
                                particles.emit_meteor_fullscreen(w, h, direction=direction, n_stars=22)
                                last_meteor = t_now
                            sweep_state[hand_label] = None
                else:
                    sweep_state[hand_label] = None

            # 꽃잎
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

        # 효과
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
