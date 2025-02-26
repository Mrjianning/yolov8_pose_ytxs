import cv2
import time
import math
import numpy as np

# 计算模块

# 计算两点之间的距离
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# # 计算三个点之间的角度
# def calculate_angle(point1, point2, point3):
#     a = calculate_distance(point1, point2)
#     b = calculate_distance(point2, point3)
#     c = calculate_distance(point1, point3)
#     return math.degrees(math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))

# 手肘角度计算
def calculate_angle(p1, p2, p3):
    # 计算两点之间的向量
    vector_a = np.array(p1) - np.array(p2)
    vector_b = np.array(p3) - np.array(p2)
    
    # 计算向量的点积
    dot_product = np.dot(vector_a, vector_b)
    
    # 计算向量的模
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    # 计算夹角的余弦值
    cos_angle = dot_product / (norm_a * norm_b)
    
    # 计算夹角的角度
    angle = np.degrees(np.arccos(cos_angle))
    
    return angle

def calculate_elbow_angle(keypoints):
    # 定义一个安全的计算角度的函数
    def safe_calculate_angle(p1, p2, p3):
        if p1 is not None and p2 is not None and p3 is not None:
            return calculate_angle(p1, p2, p3)
        else:
            return None  # 或者返回一个特定的值，表示角度无法计算
    
    # 左肘角度
    left_shoulder = keypoints[5] if len(keypoints) > 5 else None
    left_elbow = keypoints[7] if len(keypoints) > 7 else None
    left_wrist = keypoints[9] if len(keypoints) > 9 else None
    left_elbow_angle = safe_calculate_angle(left_shoulder, left_elbow, left_wrist)
    
    # 右肘角度
    right_shoulder = keypoints[6] if len(keypoints) > 6 else None
    right_elbow = keypoints[8] if len(keypoints) > 8 else None
    right_wrist = keypoints[10] if len(keypoints) > 10 else None
    right_elbow_angle = safe_calculate_angle(right_shoulder, right_elbow, right_wrist)

    return left_elbow_angle, right_elbow_angle