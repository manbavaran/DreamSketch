import numpy as np
import time

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

    # 세 손가락이 충분히 펴졌는지
    extended = 0
    for tid in [12, 16, 20]:
        tip, pip = lm.landmark[tid], lm.landmark[tid-2]
        if tip.y < pip.y - scale * 0.12:  # ← 0.09 → 0.12로 더 엄격히
            extended += 1

    return d < scale * 0.3 and extended == 3

def is_ok_released(prev_ok, curr_ok):
    return prev_ok and not curr_ok

def is_index_finger_up(multi_hand_landmarks):
    if not multi_hand_landmarks or len(multi_hand_landmarks) != 1:
        return False
    lm = multi_hand_landmarks[0]
    scale = get_hand_scale(lm)
    index_tip, index_pip = lm.landmark[8], lm.landmark[6]
    is_index_up = index_tip.y < index_pip.y - scale*0.08
    folded = 0
    for tid in [4, 12, 16, 20]:
        tip, pip = lm.landmark[tid], lm.landmark[tid-2]
        if tip.y > pip.y - scale*0.02:
            folded += 1
    return is_index_up and folded == 3

# 완전히 배타적인 양손 하트: 모든 손가락이 충분히 구부러지고(편손 NO), 엄지와 검지 끝만 매우 근접
def is_heart_gesture(multi_hand_landmarks):
    if not multi_hand_landmarks or len(multi_hand_landmarks) != 2:
        return False
    lm1, lm2 = multi_hand_landmarks
    # 모든 손가락이 구부러져야 함 (각 손의 4,8,12,16,20 tip이 pip보다 아래)
    for lm in (lm1, lm2):
        for tid in [4, 8, 12, 16, 20]:
            tip, pip = lm.landmark[tid], lm.landmark[tid-2]
            if tip.y < pip.y - 0.03:  # tip이 pip보다 아래에 있어야 (충분히 구부림)
                return False
    # 엄지 끝과 검지 끝이 서로 근접해야 함
    thumb_dist = np.linalg.norm(
        np.array([lm1.landmark[4].x, lm1.landmark[4].y]) -
        np.array([lm2.landmark[4].x, lm2.landmark[4].y]))
    index_dist = np.linalg.norm(
        np.array([lm1.landmark[8].x, lm1.landmark[8].y]) -
        np.array([lm2.landmark[8].x, lm2.landmark[8].y]))
    # 거리 임계값 엄격하게
    if thumb_dist > 0.07 or index_dist > 0.09:
        return False
    # 엄지, 검지가 서로 교차/포개어져 있으면 True
    return True



def is_palm_open(lm):
    scale = get_hand_scale(lm)
    # 4,8,12,16,20 tip이 pip보다 많이 위: 완전히 편 상태
    return all(
        lm.landmark[tid].y < lm.landmark[tid-2].y - scale*0.09
        for tid in [4,8,12,16,20]
    )



def update_sweep_traj(lm, prev_traj, palm_open):
    x, y = lm.landmark[9].x, lm.landmark[9].y
    t = time.time()
    if not palm_open:
        return []
    if not prev_traj or len(prev_traj) == 0 or t - prev_traj[-1][2] > 0.8:
        return [(x, y, t)]
    if np.hypot(x - prev_traj[-1][0], y - prev_traj[-1][1]) > 0.005:
        prev_traj.append((x, y, t))
    prev_traj = [pt for pt in prev_traj if t - pt[2] < 1.3]
    return prev_traj

def sweep_traj_distance(traj):
    if len(traj) < 2:
        return 0
    dist = 0
    for i in range(1, len(traj)):
        dist += np.hypot(traj[i][0] - traj[i-1][0], traj[i][1] - traj[i-1][1])
    return dist

def sweep_direction(traj):
    if len(traj) < 2:
        return None
    dx = traj[-1][0] - traj[0][0]
    dy = traj[-1][1] - traj[0][1]
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    elif abs(dy) > abs(dx):
        return "down" if dy > 0 else "up"
    return None


def is_one_hand_heart(multi_hand_landmarks):
    if not multi_hand_landmarks or len(multi_hand_landmarks) != 1:
        return False
    lm = multi_hand_landmarks[0]
    scale = get_hand_scale(lm)

    # Tip 좌표
    thumb = np.array([lm.landmark[4].x, lm.landmark[4].y])
    index = np.array([lm.landmark[8].x, lm.landmark[8].y])
    dist = np.linalg.norm(thumb - index)

    # 벡터 각도 (V자 정도 허용)
    thumb_base = np.array([lm.landmark[2].x, lm.landmark[2].y])
    index_base = np.array([lm.landmark[5].x, lm.landmark[5].y])
    v1 = thumb - thumb_base
    v2 = index - index_base
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
    angle_deg = np.degrees(angle)

    # 접힌 손가락 확인
    folded = 0
    for tid in [12, 16, 20]:
        if lm.landmark[tid].y > lm.landmark[tid - 2].y + scale * 0.03:
            folded += 1

    # 조건: 손가락 접힘 + 엄지검지 각도 적당함 + 거리 허용
    return folded >= 2 and dist < scale * 0.28 and 25 <= angle_deg <= 95



