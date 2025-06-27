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

    PETAL_COOLTIME = 1.0
    METEOR_COOLTIME = 1.5
    HEART_COOLTIME = 1.0
    FINISH_HOLD = 0.4
    FINISH_FADE = 0.7

    last_petal = 0
    last_meteor = 0
    last_heart = 0

    cv2.namedWindow("DreamSketch", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("DreamSketch", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fullscreen = True
    
    meteor_dir = "dr"  # 초기값: 오른쪽 아래로
    
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

        if effect_mode == "idle":
            # ✅ 각 손마다 개별로 fist → palm 전환 감지하여 꽃잎 발사
            for hand in hands_info:
                label = hand["label"]
                lm = hand["lm"]
                flipped, _ = is_fist_palm_flip_seq(lm, prev_flip_state[label])
                if flipped and t_now - last_petal > PETAL_COOLTIME:
                    effect_mode = "petal"
                    mode_t0 = t_now
                    last_petal = t_now
                    break
                    
            for hand in hands_info:
                lm = hand["lm"]
                label = hand["label"]
                palm_open = is_palm_open(lm)
                sweep_traj_dict[label] = update_sweep_traj(lm, sweep_traj_dict[label], palm_open)
                if palm_open and sweep_traj_distance(sweep_traj_dict[label]) > 0.33 and t_now - last_meteor > METEOR_COOLTIME:
                    dir = sweep_direction(sweep_traj_dict[label])
                    if dir in ("right", "left"):
                        effect_mode = "meteor"
                        meteor_dir = dir  # <-- 방향 저장
                        mode_t0 = t_now
                        last_meteor = t_now             
                        
            if len(hands_info) == 2:
                if is_heart_gesture([hands_info[0]["lm"], hands_info[1]["lm"]]) and t_now - last_heart > HEART_COOLTIME:
                    effect_mode = "heart"
                    mode_t0 = t_now
                    last_heart = t_now
                    
            if len(hands_info) == 1 and curr_ok:
                effect_mode = "ready"
                mode_t0 = t_now
        elif effect_mode == "ready":
            if len(hands_info) == 1 and is_index_finger_up([hands_info[0]["lm"]]):
                effect_mode = "draw"
                mode_t0 = t_now
                traj.clear()
            elif t_now - mode_t0 > 2.2:
                effect_mode = "idle"
        elif effect_mode == "draw":
            if len(hands_info) == 1:
                lm = hands_info[0]["lm"]
                ix = int(lm.landmark[8].x * w)
                iy = int(lm.landmark[8].y * h)
                traj.add_point((ix, iy))
            if curr_ok and t_now - mode_t0 > 0.25:
                effect_mode = "finish"
                mode_t0 = t_now
        elif effect_mode == "finish":
            dt = t_now - mode_t0
            if dt < FINISH_HOLD:
                pass
            elif dt < FINISH_HOLD + FINISH_FADE:
                pass
            else:
                effect_mode = "idle"
                traj.clear()
        elif effect_mode == "petal":
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
            particles.emit_meteor_rain(w, h, direction=meteor_dir, n=22)
            effect_mode = "idle"    
            
        elif effect_mode == "heart":
            if len(hands_info) == 2:
                lm1 = hands_info[0]["lm"]
                lm2 = hands_info[1]["lm"]
                cx = int(w * (lm1.landmark[9].x + lm2.landmark[9].x) / 2)
                cy = int(h * (lm1.landmark[9].y + lm2.landmark[9].y) / 2)
                particles.emit(cx, cy, n=54, kind="heart")
            effect_mode = "idle"

        if len(hands_info) > 0:
            for hand in hands_info:
                label = hand["label"]
                _, prev_flip_state[label] = is_fist_palm_flip_seq(hand["lm"], prev_flip_state[label])

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

        key = cv2.waitKey(1)
        if key == 27: break
        if key == ord('f') or key == ord('F') or key == 0x7a:  # F11 등
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty("DreamSketch", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty("DreamSketch", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
