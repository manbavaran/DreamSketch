# gesture.py

import numpy as np

def get_hand_scale(lm):
    # 손 전체 크기(중앙-중지tip, 엄지tip-소지tip 거리 등)로 동적 임계값 보정
    palm_base = np.array([lm.landmark[0].x, lm.landmark[0].y])
    index_tip = np.array([lm.landmark[8].x, lm.landmark[8].y])
    pinky_tip = np.array([lm.landmark[20].x, lm.landmark[20].y])
    scale = np.linalg.norm(index_tip - pinky_tip)
    return max(scale, 1e-3)

def is_ok_sign(multi_hand_landmarks):
    if not multi_hand_landmarks:
        return False
    for lm in multi_hand_landmarks:
        scale = get_hand_scale(lm)
        thumb_tip = np.array([lm.landmark[4].x, lm.landmark[4].y])
        index_tip = np.array([lm.landmark[8].x, lm.landmark[8].y])
        d = np.linalg.norm(thumb_tip - index_tip)
        if d < scale * 0.4:  # palm 크기 기반 임계값
            return True
    return False

def is_index_finger_up(multi_hand_landmarks):
    if not multi_hand_landmarks:
        return False
    for lm in multi_hand_landmarks:
        index_tip, index_pip = lm.landmark[8], lm.landmark[6]
        is_index_up = index_tip.y < index_pip.y
        is_middle_down = lm.landmark[12].y > lm.landmark[10].y
        is_ring_down = lm.landmark[16].y > lm.landmark[14].y
        is_pinky_down = lm.landmark[20].y > lm.landmark[18].y
        # 엄지는 완전히 펴지 않아도 되므로 제외(너무 빡빡X)
        if is_index_up and is_middle_down and is_ring_down and is_pinky_down:
            return True
    return False

def is_fist(multi_hand_landmarks):
    if not multi_hand_landmarks:
        return False
    for lm in multi_hand_landmarks:
        # 네 손가락 tip이 모두 pip 아래(접힘)
        if all(lm.landmark[tid].y > lm.landmark[tid-2].y for tid in [8,12,16,20]):
            # 엄지도 손바닥 쪽(엄지tip이 index_mcp보다 아래)
            if lm.landmark[4].y > lm.landmark[5].y:
                return True
    return False

def is_hand_open(multi_hand_landmarks):
    if not multi_hand_landmarks:
        return False
    for lm in multi_hand_landmarks:
        # 네 손가락 모두 tip이 pip 위(펴짐)
        if all(lm.landmark[tid].y < lm.landmark[tid-2].y for tid in [8,12,16,20]):
            # 엄지도 어느정도 벌려져 있으면 open
            if lm.landmark[4].x < lm.landmark[3].x or lm.landmark[4].x > lm.landmark[3].x:
                return True
    return False

def is_heart_gesture(multi_hand_landmarks):
    # 양손 필요, 각각의 index_tip, thumb_tip이 가까워야함(좌우 대칭)
    if not multi_hand_landmarks or len(multi_hand_landmarks) != 2:
        return False
    lm1, lm2 = multi_hand_landmarks
    scale1 = get_hand_scale(lm1)
    scale2 = get_hand_scale(lm2)
    idx_tip1 = np.array([lm1.landmark[8].x, lm1.landmark[8].y])
    thb_tip1 = np.array([lm1.landmark[4].x, lm1.landmark[4].y])
    idx_tip2 = np.array([lm2.landmark[8].x, lm2.landmark[8].y])
    thb_tip2 = np.array([lm2.landmark[4].x, lm2.landmark[4].y])
    # index1~thumb2, index2~thumb1 각각 가까우면 하트
    d1 = np.linalg.norm(idx_tip1 - thb_tip2)
    d2 = np.linalg.norm(idx_tip2 - thb_tip1)
    if d1 < (scale1+scale2)*0.3 and d2 < (scale1+scale2)*0.3:
        return True
    return False
