import numpy as np

def get_hand_scale(lm):
    index_tip = np.array([lm.landmark[8].x, lm.landmark[8].y])
    pinky_tip = np.array([lm.landmark[20].x, lm.landmark[20].y])
    scale = np.linalg.norm(index_tip - pinky_tip)
    return max(scale, 1e-4)

def is_ok_sign(multi_hand_landmarks):
    if not multi_hand_landmarks or len(multi_hand_landmarks) != 1:
        return False
    lm = multi_hand_landmarks[0]
    scale = get_hand_scale(lm)
    thumb_tip = np.array([lm.landmark[4].x, lm.landmark[4].y])
    index_tip = np.array([lm.landmark[8].x, lm.landmark[8].y])
    d = np.linalg.norm(thumb_tip - index_tip)
    is_middle_down = lm.landmark[12].y > lm.landmark[10].y + scale*0.05
    is_ring_down = lm.landmark[16].y > lm.landmark[14].y + scale*0.05
    is_pinky_down = lm.landmark[20].y > lm.landmark[18].y + scale*0.05
    if d < scale * 0.47 and is_middle_down and is_ring_down and is_pinky_down:
        return True
    return False

def is_index_finger_up(multi_hand_landmarks):
    if not multi_hand_landmarks or len(multi_hand_landmarks) != 1:
        return False
    lm = multi_hand_landmarks[0]
    scale = get_hand_scale(lm)
    index_tip, index_pip = lm.landmark[8], lm.landmark[6]
    is_index_up = index_tip.y < index_pip.y - scale*0.10
    down_cnt = 0
    for tid in [12, 16, 20]:
        tip = lm.landmark[tid]
        pip = lm.landmark[tid-2]
        if tip.y > pip.y - scale*0.03:
            down_cnt += 1
    if is_index_up and down_cnt >= 2:
        return True
    return False

def is_heart_gesture(multi_hand_landmarks):
    if not multi_hand_landmarks or len(multi_hand_landmarks) != 2:
        return False
    lm1, lm2 = multi_hand_landmarks
    scale1 = get_hand_scale(lm1)
    scale2 = get_hand_scale(lm2)
    avg_scale = (scale1 + scale2) / 2
    idx_tip1 = np.array([lm1.landmark[8].x, lm1.landmark[8].y])
    thb_tip2 = np.array([lm2.landmark[4].x, lm2.landmark[4].y])
    idx_tip2 = np.array([lm2.landmark[8].x, lm2.landmark[8].y])
    thb_tip1 = np.array([lm1.landmark[4].x, lm1.landmark[4].y])
    d1 = np.linalg.norm(idx_tip1 - thb_tip2)
    d2 = np.linalg.norm(idx_tip2 - thb_tip1)
    if d1 < avg_scale * 0.32 and d2 < avg_scale * 0.32:
        return True
    return False

def is_front_fist(lm):
    scale = get_hand_scale(lm)
    is_fist = all(
        lm.landmark[tid].y > lm.landmark[tid-2].y + scale*0.04
        for tid in [8,12,16,20]
    )
    wrist_z = lm.landmark[0].z
    mtip_z = lm.landmark[12].z
    return is_fist and (wrist_z < mtip_z)

def is_palm_open(lm):
    scale = get_hand_scale(lm)
    is_open = all(
        lm.landmark[tid].y < lm.landmark[tid-2].y - scale*0.09
        for tid in [8,12,16,20]
    )
    wrist_z = lm.landmark[0].z
    mtip_z = lm.landmark[12].z
    return is_open and (wrist_z > mtip_z)

def is_fist_palm_flip_seq(lm, prev_state):
    if is_front_fist(lm):
        now_state = "front_fist"
    elif is_palm_open(lm):
        now_state = "palm_open"
    else:
        now_state = None
    if prev_state == "front_fist" and now_state == "palm_open":
        return True, now_state
    return False, now_state
