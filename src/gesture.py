import numpy as np

def get_hand_scale(lm):
    index_tip = np.array([lm.landmark[8].x, lm.landmark[8].y])
    pinky_tip = np.array([lm.landmark[20].x, lm.landmark[20].y])
    return max(np.linalg.norm(index_tip - pinky_tip), 1e-4)

def is_ok_sign(multi_hand_landmarks):
    """엄지-검지 붙고 나머지 세 손가락은 '펴져있는' OK 사인."""
    if not multi_hand_landmarks or len(multi_hand_landmarks) != 1:
        return False
    lm = multi_hand_landmarks[0]
    scale = get_hand_scale(lm)
    thumb_tip = np.array([lm.landmark[4].x, lm.landmark[4].y])
    index_tip = np.array([lm.landmark[8].x, lm.landmark[8].y])
    d = np.linalg.norm(thumb_tip - index_tip)
    # 중지, 약지, 소지 '펴짐'
    extended_count = 0
    for tid in [12, 16, 20]:
        tip = lm.landmark[tid]
        pip = lm.landmark[tid-2]
        if tip.y < pip.y - scale * 0.05:  # 충분히 펴져있을 때만 인정
            extended_count += 1
    if d < scale * 0.5 and extended_count == 3:
        return True
    return False

def is_ok_released(prev_ok, curr_ok):
    """OK 사인에서 OK 해제된 순간만 True"""
    return prev_ok and not curr_ok

def is_index_finger_up(multi_hand_landmarks):
    """검지 손가락만 펴고 나머지는 접은 상태"""
    if not multi_hand_landmarks or len(multi_hand_landmarks) != 1:
        return False
    lm = multi_hand_landmarks[0]
    scale = get_hand_scale(lm)
    # 검지만 확실히 핀 상태
    index_tip, index_pip = lm.landmark[8], lm.landmark[6]
    is_index_up = index_tip.y < index_pip.y - scale * 0.07
    folded_count = 0
    for tid in [4, 12, 16, 20]:
        tip, pip = lm.landmark[tid], lm.landmark[tid-2]
        if tip.y > pip.y - scale * 0.02:  # tip이 pip보다 충분히 아래
            folded_count += 1
    if is_index_up and folded_count == 3:
        return True
    return False

def is_heart_gesture(multi_hand_landmarks):
    """양손 하트: 손가락 모두 펴고, 각 손가락 끝끼리 서로 매우 가까움"""
    if not multi_hand_landmarks or len(multi_hand_landmarks) != 2:
        return False
    lm1, lm2 = multi_hand_landmarks
    # 각 손가락 모두 '펴져 있음' 확인
    for lm in (lm1, lm2):
        for tid in [4, 8, 12, 16, 20]:
            tip = lm.landmark[tid]
            pip = lm.landmark[tid-2]
            if tip.y > pip.y - 0.03:
                return False
    # 손가락 끝끼리의 거리 체크
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
        if np.linalg.norm(pt1 - pt2) > avg_scale * 0.21:
            return False
    if abs(lm1.landmark[0].y - lm2.landmark[0].y) > 0.13:
        return False
    return True

def is_front_fist(lm):
    scale = get_hand_scale(lm)
    is_fist = all(
        lm.landmark[tid].y > lm.landmark[tid-2].y + scale*0.025
        for tid in [8,12,16,20]
    )
    return is_fist

def is_palm_open(lm):
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
