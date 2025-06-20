import numpy as np

def get_hand_scale(lm):
    # 검지 끝과 새끼 끝 거리(넓이)로 손의 상대적 크기 측정
    index_tip = np.array([lm.landmark[8].x, lm.landmark[8].y])
    pinky_tip = np.array([lm.landmark[20].x, lm.landmark[20].y])
    scale = np.linalg.norm(index_tip - pinky_tip)
    return max(scale, 1e-4)

def is_ok_sign(multi_hand_landmarks):
    if not multi_hand_landmarks:
        return False
    for lm in multi_hand_landmarks:
        scale = get_hand_scale(lm)
        thumb_tip = np.array([lm.landmark[4].x, lm.landmark[4].y])
        index_tip = np.array([lm.landmark[8].x, lm.landmark[8].y])
        d = np.linalg.norm(thumb_tip - index_tip)
        # 손 크기 대비 좀 더 넉넉하게(기존 0.35 → 0.45)
        if d < scale * 0.45:
            return True
    return False

def is_index_finger_up(multi_hand_landmarks):
    if not multi_hand_landmarks:
        return False
    for lm in multi_hand_landmarks:
        scale = get_hand_scale(lm)
        index_tip, index_pip = lm.landmark[8], lm.landmark[6]
        # 검지가 pip보다 scale*0.09 이상 위에 있으면 OK
        is_index_up = index_tip.y < index_pip.y - scale*0.09
        down_cnt = 0
        for tid in [12, 16, 20]:
            pip = lm.landmark[tid-2]
            tip = lm.landmark[tid]
            # 손 크기 기준으로 좀 더 여유롭게
            if tip.y > pip.y - scale*0.02:
                down_cnt += 1
        # 3개 중 2개만 내려가도 OK
        if is_index_up and down_cnt >= 2:
            return True
    return False

def is_heart_gesture(multi_hand_landmarks):
    if not multi_hand_landmarks or len(multi_hand_landmarks) != 2:
        return False
    lm1, lm2 = multi_hand_landmarks
    scale1 = get_hand_scale(lm1)
    scale2 = get_hand_scale(lm2)
    idx_tip1 = np.array([lm1.landmark[8].x, lm1.landmark[8].y])
    thb_tip1 = np.array([lm1.landmark[4].x, lm1.landmark[4].y])
    idx_tip2 = np.array([lm2.landmark[8].x, lm2.landmark[8].y])
    thb_tip2 = np.array([lm2.landmark[4].x, lm2.landmark[4].y])
    d1 = np.linalg.norm(idx_tip1 - thb_tip2)
    d2 = np.linalg.norm(idx_tip2 - thb_tip1)
    # 손 크기별 평균값 기준을 활용해 여유롭게 (0.3)
    avg_scale = (scale1 + scale2) / 2
    if d1 < avg_scale * 0.3 and d2 < avg_scale * 0.3:
        return True
    return False

def is_vertical_sweep_gesture(lm, prev_x, frame_w, min_dx=None):
    scale = get_hand_scale(lm)
    # 손 크기에 따라 동적으로 이동 기준 설정
    dynamic_min_dx = scale * frame_w * 1.6 if min_dx is None else min_dx
    wrist = lm.landmark[0]
    mtip = lm.landmark[12]
    dy = abs(mtip.y - wrist.y)
    dx = abs(mtip.x - wrist.x)
    vertical = dy > dx * 1.7
    open_palm = all(lm.landmark[tid].y < lm.landmark[tid-2].y for tid in [8,12,16,20])
    if prev_x is not None and open_palm and vertical:
        if abs(lm.landmark[9].x * frame_w - prev_x) > dynamic_min_dx:
            return True
    return False

def is_front_fist(lm):
    # 손가락 모두 접힘 + 손등이 카메라 쪽 (wrist z < middle tip z)
    is_fist = all(lm.landmark[tid].y > lm.landmark[tid-2].y for tid in [8,12,16,20])
    wrist_z = lm.landmark[0].z
    mtip_z = lm.landmark[12].z
    return is_fist and (wrist_z < mtip_z)

def is_palm_open(lm):
    # 손가락 모두 펴짐 + 손바닥이 카메라 쪽 (wrist z > middle tip z)
    is_open = all(lm.landmark[tid].y < lm.landmark[tid-2].y for tid in [8,12,16,20])
    wrist_z = lm.landmark[0].z
    mtip_z = lm.landmark[12].z
    return is_open and (wrist_z > mtip_z)

def is_fist_palm_flip_seq(lm, prev_state):
    # prev_state: "front_fist" or "palm_open" or None
    if is_front_fist(lm):
        now_state = "front_fist"
    elif is_palm_open(lm):
        now_state = "palm_open"
    else:
        now_state = None
    # 주먹(손등) → 손바닥 전이
    if prev_state == "front_fist" and now_state == "palm_open":
        return True, now_state
    return False, now_state
