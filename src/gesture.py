import numpy as np

def get_hand_scale(lm):
    index_tip = np.array([lm.landmark[8].x, lm.landmark[8].y])
    pinky_tip = np.array([lm.landmark[20].x, lm.landmark[20].y])
    return max(np.linalg.norm(index_tip - pinky_tip), 1e-4)

def is_ok_sign(multi_hand_landmarks):
    if not multi_hand_landmarks or len(multi_hand_landmarks) != 1:
        return False
    lm = multi_hand_landmarks[0]
    scale = get_hand_scale(lm)
    thumb_tip = np.array([lm.landmark[4].x, lm.landmark[4].y])
    index_tip = np.array([lm.landmark[8].x, lm.landmark[8].y])
    d = np.linalg.norm(thumb_tip - index_tip)
    # 엄지-검지 거리만 체크, 나머지 손가락은 좀 펴져 있어도 됨
    if d < scale * 0.55:
        return True
    return False

def is_index_finger_up(multi_hand_landmarks):
    if not multi_hand_landmarks or len(multi_hand_landmarks) != 1:
        return False
    lm = multi_hand_landmarks[0]
    index_tip, index_pip = lm.landmark[8], lm.landmark[6]
    # 검지만 확실히 핀 상태, 나머지는 대충 접혀있으면 됨
    is_index_up = index_tip.y < index_pip.y - 0.06
    fingers_folded = 0
    for tid in [4, 12, 16, 20]:
        tip, pip = lm.landmark[tid], lm.landmark[tid-2]
        if tip.y > pip.y - 0.01:
            fingers_folded += 1
    if is_index_up and fingers_folded >= 3:
        return True
    return False

def is_heart_gesture(multi_hand_landmarks):
    if not multi_hand_landmarks or len(multi_hand_landmarks) != 2:
        return False
    lm1, lm2 = multi_hand_landmarks
    # 양손 모두 손가락을 확실히 핀 상태만 인정
    for lm in (lm1, lm2):
        for tid in [4, 8, 12, 16, 20]:
            tip = lm.landmark[tid]
            pip = lm.landmark[tid-2]
            if tip.y > pip.y - 0.02:  # tip이 pip보다 0.02 이상 위에 있어야 함
                return False
    # 손가락 끝끼리 거리 체크(약간 더 엄격하게)
    scale1 = np.linalg.norm(
        np.array([lm1.landmark[8].x, lm1.landmark[8].y]) -
        np.array([lm1.landmark[20].x, lm1.landmark[20].y]))
    scale2 = np.linalg.norm(
        np.array([lm2.landmark[8].x, lm2.landmark[8].y]) -
        np.array([lm2.landmark[20].x, lm2.landmark[20].y]))
    avg_scale = (scale1 + scale2) / 2.0
    for fid in [4,8,12,16,20]:
        pt1 = np.array([lm1.landmark[fid].x, lm1.landmark[fid].y])
        pt2 = np.array([lm2.landmark[fid].x, lm2.landmark[fid].y])
        if np.linalg.norm(pt1 - pt2) > avg_scale * 0.20:  # 엄격하게(20%)
            return False
    # 손목 높이도 유사해야 함
    if abs(lm1.landmark[0].y - lm2.landmark[0].y) > 0.10:
        return False
    return True

def is_front_fist(lm):
    # 모든 손가락 tip이 pip보다 아래(y 기준)면 주먹
    scale = get_hand_scale(lm)
    is_fist = all(
        lm.landmark[tid].y > lm.landmark[tid-2].y + scale*0.025
        for tid in [8,12,16,20]
    )
    return is_fist

def is_palm_open(lm):
    # 모든 손가락 tip이 pip보다 위(y 기준)면 손바닥
    scale = get_hand_scale(lm)
    is_open = all(
        lm.landmark[tid].y < lm.landmark[tid-2].y - scale*0.04
        for tid in [8,12,16,20]
    )
    return is_open



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
