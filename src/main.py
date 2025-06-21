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
    effect_mode = "idle"
    mode_t0 = time.time()
    prev_flip_state = {"left": None, "right": None}
    prev_ok_state = False
    sweep_traj_dict = {"left": [], "right": []}

    cv2.namedWindow("DreamSketch", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("DreamSketch", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 쿨타임/길이
    PETAL_COOLTIME = 1.0
    METEOR_COOLTIME = 1.5
    HEART_COOLTIME = 1.0
    FINISH_HOLD = 0.4
    FINISH_FADE = 0.7

    last_petal = 0
    last_meteor = 0
    last_heart = 0

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

        # mode 진입 분기(순서 매우 중요!)
        if effect_mode == "idle":
            # 1. 양손 플립 → 동시 petal
            if len(hands_info) == 2:
                left_flip, _ = is_fist_palm_flip_seq(hands_info[0]["lm"], prev_flip_state["left"])
                right_flip, _ = is_fist_palm_flip_seq(hands_info[1]["lm"], prev_flip_state["right"])
                if left_flip and right_flip and t_now - last_petal > PETAL_COOLTIME:
                    effect_mode = "petal"
                    mode_t0 = t_now
                    last_petal = t_now
            # 2. 손날 스윕 → meteor
            for hand in hands_info:
                lm = hand["lm"]
                label = hand["label"]
                palm_open = is_palm_open(lm)
                sweep_traj_dict[label] = update_sweep_traj(lm, sweep_traj_dict[label], palm_open)
                if palm_open and sweep_traj_distance(sweep_traj_dict[label]) > 0.33 and t_now - last_meteor > METEOR_COOLTIME:
                    dir = sweep_direction(sweep_traj_dict[label])
                    if dir in ("right", "left"):
                        effect_mode = "meteor"
                        mode_t0 = t_now
                        last_meteor = t_now
            # 3. 양손 하트
            if len(hands_info) == 2:
                if is_heart_gesture([hands_info[0]["lm"], hands_info[1]["lm"]]) and t_now - last_heart > HEART_COOLTIME:
                    effect_mode = "heart"
                    mode_t0 = t_now
                    last_heart = t_now
            # 4. draw 모드 진입(OK→index 연계)
            if len(hands_info) == 1 and curr_ok:
                effect_mode = "draw"
                mode_t0 = t_now
                traj.clear()

        elif effect_mode == "petal":
            # 양손 위치 각각 burst
            for hand in hands_info:
                lm = hand["lm"]
                label = hand["label"]
                idx = int(lm.landmark[9].x * w), int(lm.landmark[9].y * h)
                if label == "left":
                    particles.emit_flower_burst(idx[0], idx[1], kind="rose", n=60)
                else:
                    particles.emit_flower_burst(idx[0], idx[1], kind="sakura", n=60)
            effect_mode = "idle"

        elif effect_mode == "meteor":
            # 유성우 전체 화면 커버
            direction = "dr"
            particles.emit_meteor_rain(w, h, direction=direction, n=22)
            effect_mode = "idle"

        elif effect_mode == "heart":
            # 하트
            particles.emit(w//2, h//2, n=54, kind="heart")
            effect_mode = "idle"

        elif effect_mode == "draw":
            if len(hands_info) == 1:
                lm = hands_info[0]["lm"]
                ix = int(lm.landmark[8].x * w)
                iy = int(lm.landmark[8].y * h)
                traj.add_point((ix, iy))
            # draw모드에서는 ok sign 외에는 finish 안 됨!
            if curr_ok and t_now - mode_t0 > 0.25:
                effect_mode = "finish"
                mode_t0 = t_now

        elif effect_mode == "finish":
            # finish: hold→fade→idle (매우 짧게)
            dt = t_now - mode_t0
            if dt < FINISH_HOLD:
                pass
            elif dt < FINISH_HOLD + FINISH_FADE:
                pass
            else:
                effect_mode = "idle"
                traj.clear()

        # flip state update (항상 마지막에!)
        if len(hands_info) > 0:
            for hand in hands_info:
                label = hand["label"]
                _, prev_flip_state[label] = is_fist_palm_flip_seq(hand["lm"], prev_flip_state[label])

        # 렌더링
        out = frame.copy()
        if effect_mode == "draw":
            out = traj.draw(out, brighten=0.0, fade_out=False)
        elif effect_mode == "finish":
            dt = time.time() - mode_t0
            if dt < FINISH_HOLD:
                out = traj.draw(out, brighten=1.0, fade_out=False)
            elif dt < FINISH_HOLD + FINISH_FADE:
                alpha = max(0, 1.0 - (dt-FINISH_HOLD)/FINISH_FADE)
                out = traj.draw(out, brighten=alpha, fade_out=True)
        out = particles.update_and_draw(out)

        cv2.putText(out, f"Mode: {effect_mode}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 180, 255), 2, cv2.LINE_AA)
        cv2.imshow("DreamSketch", out)
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
