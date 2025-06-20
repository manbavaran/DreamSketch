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
    scale1 = np.linalg.norm(lm1.landmark[8].x - lm1.landmark[20].x)
    scale2 = np.linalg.norm(lm2.landmark[8].x - lm2.landmark[20].x)
    avg_scale = (scale1 + scale2)/2
    cx1, cy1 = lm1.landmark[0].x, lm1.landmark[0].y
    cx2, cy2 = lm2.landmark[0].x, lm2.landmark[0].y
    center_x = (cx1+cx2)/2
    y_diff = abs(cy1-cy2)
    if not (0.25 < center_x < 0.75): return False
    if y_diff > 0.07: return False
    v1 = np.array([lm1.landmark[8].x - lm1.landmark[4].x, lm1.landmark[8].y - lm1.landmark[4].y])
    v2 = np.array([lm2.landmark[8].x - lm2.landmark[4].x, lm2.landmark[8].y - lm2.landmark[4].y])
    angle = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8))
    if angle > 0.35: return False
    d1 = np.linalg.norm(np.array([lm1.landmark[8].x, lm1.landmark[8].y]) - np.array([lm1.landmark[4].x, lm1.landmark[4].y]))
    d2 = np.linalg.norm(np.array([lm2.landmark[8].x, lm2.landmark[8].y]) - np.array([lm2.landmark[4].x, lm2.landmark[4].y]))
    cross1 = np.linalg.norm(np.array([lm1.landmark[8].x, lm1.landmark[8].y]) - np.array([lm2.landmark[4].x, lm2.landmark[4].y]))
    cross2 = np.linalg.norm(np.array([lm2.landmark[8].x, lm2.landmark[8].y]) - np.array([lm1.landmark[4].x, lm1.landmark[4].y]))
    if d1 > scale1*0.53 or d2 > scale2*0.53 or cross1 > avg_scale*0.42 or cross2 > avg_scale*0.42:
        return False
    fingers1 = [lm1.landmark[tid].y > lm1.landmark[tid-2].y+scale1*0.012 for tid in [12, 16, 20]]
    fingers2 = [lm2.landmark[tid].y > lm2.landmark[tid-2].y+scale2*0.012 for tid in [12, 16, 20]]
    if sum(fingers1) < 2 or sum(fingers2) < 2:
        return False
    return True

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
