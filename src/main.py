import cv2
import mediapipe as mp
import time
from gesture import (
    is_ok_sign, is_ok_released, is_index_finger_up, is_heart_gesture,
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
    OK_HOLD_SEC = 0.2

    ok_just_released_time = None
    ok_released_window = 2.0  # OK 해제 후 2초 안에 index up 인식 시 draw 진입

    prev_ok_state = False

    ready_since = None
    READY_TIMEOUT = 4.0

    sweep_state = {"left": None, "right": None}
    SWEEP_TRIGGER_TIME = 1.5
    SWEEP_MIN_DIST = 0.3

    last_gesture_name = ""
    last_gesture_time = 0
    GESTURE_DISPLAY_DURATION = 2.0

    cv2.namedWindow("DreamSketch", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("DreamSketch", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        t_now = time.time()

        # --- MediaPipe Handedness 정보 동기화 ---
        hands_info = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for lm, handness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handness.classification[0].label.lower()  # 'left' or 'right'
                hands_info.append({"lm": lm, "label": label})

        # --- OK 사인 인식, 해제 감지 ---
        curr_ok = len(hands_info) == 1 and is_ok_sign([hands_info[0]["lm"]])
        ok_released = is_ok_released(prev_ok_state, curr_ok)
        prev_ok_state = curr_ok

        # --- 상태머신 ---
        if mode == "idle":
            if curr_ok:
                if ok_since is None:
                    ok_since = t_now
                elif t_now - ok_since > OK_HOLD_SEC:
                    mode = "ready"
                    ready_since = t_now
                    ok_since = None
                    last_gesture_name = "OK Sign"
                    last_gesture_time = t_now
            else:
                ok_since = None
                ready_since = None
            # OK 사인을 해제한 시점 기록 (손가락 바뀔 때)
            if ok_released:
                ok_just_released_time = t_now
            # OK 해제 후 일정 시간 이내에 index up → draw 진입
            if (ok_just_released_time is not None and 
                t_now - ok_just_released_time < ok_released_window and
                len(hands_info) == 1 and is_index_finger_up([hands_info[0]["lm"]])):
                mode = "draw"
                traj.active = True
                ok_just_released_time = None
                last_gesture_name = "Index Up"
                last_gesture_time = t_now

        elif mode == "ready":
            # (ready 상태에서는 index up 진입 불필요, OK 해제 감지만 해도 됨)
            if ok_released:
                ok_just_released_time = t_now
            if (ok_just_released_time is not None and 
                t_now - ok_just_released_time < ok_released_window and
                len(hands_info) == 1 and is_index_finger_up([hands_info[0]["lm"]])):
                mode = "draw"
                traj.active = True
                ok_just_released_time = None
                last_gesture_name = "Index Up"
                last_gesture_time = t_now
            elif ready_since and (t_now - ready_since > READY_TIMEOUT):
                mode = "idle"
                ready_since = None
                ok_just_released_time = None
            elif len(hands_info) == 0:
                ready_since = None
                ok_just_released_time = None

        elif mode == "draw":
            if curr_ok:
                mode = "finish"
                fade_stage = "brighten"
                fade_t0 = time.time()
                traj.active = False
                last_gesture_name = "OK Sign (Finish)"
                last_gesture_time = t_now

        # --- 글씨 쓰기: 인덱스 손가락 이동 경로 저장 ---
        if mode == "draw" and len(hands_info) == 1:
            lm = hands_info[0]["lm"]
            ix = int(lm.landmark[8].x * w)
            iy = int(lm.landmark[8].y * h)
            traj.add_point((ix, iy))
            particles.emit(ix, iy, n=1, kind="star")

        # --- idle 모드: 이펙트 전용 ---
        if mode == "idle":
            # --- (1) 하트 ---
            if len(hands_info) == 2:
                if is_heart_gesture([hands_info[0]["lm"], hands_info[1]["lm"]]) and (t_now - last_heart > 2):
                    particles.emit(w//2, h//2, n=40, kind="heart")
                    last_heart = t_now
                    last_gesture_name = "Heart"
                    last_gesture_time = t_now

            # --- (2) 손날 스윕 ---
            for hand in hands_info:
                lm = hand["lm"]
                hand_label = hand["label"]
                open_palm = all(
                    lm.landmark[tid].y < lm.landmark[tid-2].y - 0.02
                    for tid in [8,12,16,20]
                )
                if open_palm:
                    now = time.time()
                    if sweep_state[hand_label] is None:
                        sweep_state[hand_label] = {
                            "x_start": lm.landmark[9].x,
                            "start_time": now
                        }
                    else:
                        dx = abs(lm.landmark[9].x - sweep_state[hand_label]["x_start"])
                        dt = now - sweep_state[hand_label]["start_time"]
                        if dt > SWEEP_TRIGGER_TIME and dx > SWEEP_MIN_DIST:
                            direction = "right" if (lm.landmark[9].x - sweep_state[hand_label]["x_start"]) > 0 else "left"
                            if t_now - last_meteor > 1.5:
                                particles.emit_meteor_grid(w, h, direction=direction, n_col=17, n_row=7)
                                last_meteor = t_now
                                last_gesture_name = f"Palm Sweep ({'->' if direction == 'right' else '<-'})"
                                last_gesture_time = t_now
                            sweep_state[hand_label] = None
                else:
                    sweep_state[hand_label] = None

            # --- (3) 꽃잎(주먹→손바닥 플립) ---
            for hand in hands_info:
                lm = hand["lm"]
                hand_label = hand["label"]
                triggered, new_state = is_fist_palm_flip_seq(lm, prev_flip_state.get(hand_label))
                if triggered and (t_now - last_rose > 3.0):
                    x_pos = int(lm.landmark[9].x * w)
                    y_pos = int(lm.landmark[9].y * h)
                    if hand_label == "left":
                        particles.emit_flower_burst(x_pos, y_pos, kind="rose", n=48, spread=1.3)
                        last_gesture_name = "Petal (Left)"
                    elif hand_label == "right":
                        particles.emit_flower_burst(x_pos, y_pos, kind="sakura", n=48, spread=1.3)
                        last_gesture_name = "Petal (Right)"
                    last_rose = t_now
                    last_gesture_time = t_now
                prev_flip_state[hand_label] = new_state

        # --- 효과/렌더링 ---
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

        # --- 제스처 이름 표시 ---
        if last_gesture_name and (t_now - last_gesture_time < GESTURE_DISPLAY_DURATION):
            text = f"check: {last_gesture_name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.2
            thickness = 3
            size = cv2.getTextSize(text, font, scale, thickness)[0]
            pos = (w//2 - size[0]//2, h - 30)
            cv2.putText(out, text, pos, font, scale, (50, 220, 90), thickness, cv2.LINE_AA)

        cv2.putText(out, f"Mode: {mode}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 180, 255), 2, cv2.LINE_AA)
        cv2.imshow("DreamSketch", out)
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
