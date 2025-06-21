import cv2
import mediapipe as mp
import time
from gesture import (
    is_ok_sign, is_ok_released, is_index_finger_up, is_heart_gesture,
    is_fist_palm_flip_seq, update_sweep_traj, sweep_traj_distance,
    sweep_direction, is_palm_open
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
    last_rose = {"left": 0, "right": 0}
    prev_flip_state = {"left": None, "right": None}
    ROSE_COOLTIME = 1.1

    sweep_traj = {"left": [], "right": []}
    last_meteor = 0
    SWEEP_COOLTIME = 1.6
    sweeping_until = 0

    ok_since = None
    OK_HOLD_SEC = 0.2
    ok_just_released_time = None
    ok_released_window = 2.0
    prev_ok_state = False

    ready_since = None
    READY_TIMEOUT = 4.0

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

        hands_info = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for lm, handness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handness.classification[0].label.lower()
                hands_info.append({"lm": lm, "label": label})

        curr_ok = len(hands_info) == 1 and is_ok_sign([hands_info[0]["lm"]])
        ok_released = is_ok_released(prev_ok_state, curr_ok)
        prev_ok_state = curr_ok

        # draw 모드에서 ok 사인시 finish 전환 (fade out)
        if mode == "draw":
            # draw 모드에서는 index up, ok 사인 인식 모두 무시!
            if curr_ok:
                mode = "finish"
                fade_stage = "brighten"
                fade_t0 = time.time()
                last_gesture_name = "Finish by OK"
                last_gesture_time = t_now

        # 상태머신
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

        elif mode == "ready":
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

        # 글씨 쓰기
        if mode == "draw" and len(hands_info) == 1:
            lm = hands_info[0]["lm"]
            ix = int(lm.landmark[8].x * w)
            iy = int(lm.landmark[8].y * h)
            traj.add_point((ix, iy))
            particles.emit(ix, iy, n=1, kind="star")

        # 이펙트/idle
        if mode == "idle":
            sweeping_any = False
            for hand in hands_info:
                lm = hand["lm"]
                label = hand["label"]
                palm_open = is_palm_open(lm)
                sweep_traj[label] = update_sweep_traj(lm, sweep_traj[label], palm_open)
                if palm_open and sweep_traj_distance(sweep_traj[label]) > 0.33:
                    dir = sweep_direction(sweep_traj[label])
                    if t_now - last_meteor > SWEEP_COOLTIME:
                        if dir in ("right", "left"):
                            particles.emit_meteor_rain(w, h, direction="dr" if dir == "right" else "dl", n=20)
                            last_meteor = t_now
                            sweeping_until = t_now + SWEEP_COOLTIME
                            last_gesture_name = f"Meteor Shower ({dir})"
                            last_gesture_time = t_now
                    sweep_traj[label] = []
                    sweeping_any = True
            sweeping_any = sweeping_any or (t_now < sweeping_until)

            # sweeping 쿨타임엔 하트/꽃잎 절대 발생 금지
            if not sweeping_any:
                # 하트: 양손 완전 배타
                if len(hands_info) == 2:
                    if is_heart_gesture([hands_info[0]["lm"], hands_info[1]["lm"]]) and (t_now - last_heart > 1.0):
                        particles.emit(w//2, h//2, n=50, kind="heart")
                        last_heart = t_now
                        last_gesture_name = "Heart"
                        last_gesture_time = t_now

                # 꽃잎: 양손 동시 플립도 각각 발생
                flips = []
                for hand in hands_info:
                    lm = hand["lm"]
                    hand_label = hand["label"]
                    triggered, new_state = is_fist_palm_flip_seq(lm, prev_flip_state.get(hand_label))
                    if triggered and (t_now - last_rose[hand_label] > ROSE_COOLTIME):
                        flips.append(hand_label)
                    prev_flip_state[hand_label] = new_state
                for hand_label in flips:
                    x_pos = int(hands_info[0]["lm"].landmark[9].x * w if hand_label == hands_info[0]["label"] else hands_info[1]["lm"].landmark[9].x * w)
                    y_pos = int(hands_info[0]["lm"].landmark[9].y * h if hand_label == hands_info[0]["label"] else hands_info[1]["lm"].landmark[9].y * h)
                    if hand_label == "left":
                        particles.emit_flower_burst(x_pos, y_pos, kind="rose", n=50, spread=1.3)
                        last_gesture_name = "Petal (Left)"
                    elif hand_label == "right":
                        particles.emit_flower_burst(x_pos, y_pos, kind="sakura", n=50, spread=1.3)
                        last_gesture_name = "Petal (Right)"
                    last_rose[hand_label] = t_now
                    last_gesture_time = t_now

        # 렌더링
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
