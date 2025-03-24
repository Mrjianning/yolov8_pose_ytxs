import onnxruntime
import threading
import numpy as np
import cv2
import time
import configparser
import os
import datetime 
from core.pose_calculator import calculate_elbow_angle,is_midpoint_above_line
from utils.utils import Utils
from utils.pose_utils import PoseUtils
# 姿态检测模块


class Keypoint:
    """基于 ONNX 模型的关键点检测类"""
    ACTION_LEVELS = ["优秀", "良好", "一般",""]  # 动作等级定义
    CONF_THRESHOLD = 0.7  # 置信度阈值
    IOU_THRESHOLD = 0.6   # NMS 的 IoU 阈值
    COUNT_TIME_THRESHOLD = 1.0  # 计数时间间隔阈值（秒）
    IMPROVE_LIST = []  # 动作改进建议

    def __init__(self, modelpath):
        """初始化 Keypoint 检测器"""
        print("==========初始化==========")
        
        # 读取配置文件
        self.config=Utils.load_ini_config("./config.ini")

        # 获取配置项，并提供默认值
        try:
            # ActionLevel 节
            improve_1 = self.config.get('ActionLevel', 'improve_1', fallback='')
            improve_2 = self.config.get('ActionLevel', 'improve_2', fallback='')
            self.IMPROVE_LIST.append(improve_1)
            self.IMPROVE_LIST.append(improve_2)

            # Settings 节
            self.is_save_image = self.config.get('Settings', 'is_save_image', fallback=False) 
            self.save_image_path = self.config.get('Settings', 'save_image_path', fallback="./data/saveImages/") 
            train_ratio = self.config.getfloat('Settings', 'train_ratio', fallback=0.8)   # 默认值 0.8
            debug = self.config.getboolean('Settings', 'debug', fallback=False)           # 默认值 False

            print("==========配置信息==========")
            print(f"self.is_save_image: {self.is_save_image}")
            print(f"improve_1: {improve_1}")
            print(f"improve_2: {improve_2}")
            print(f"train_ratio: {train_ratio}")
            print(f"debug: {debug}")

        except ValueError as e:
            raise ValueError(f"配置项类型转换失败: {e}")
        except Exception as e:
            raise ValueError(f"读取配置项失败: {e}")

        # 加载模型
        self.session = onnxruntime.InferenceSession(modelpath, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.label_name = self.session.get_outputs()[0].name
        self.last_count_time = 0.0  # 上次计数时间戳
        self.was_above = False      # 中点是否在上方的状态
        self.action_levels=""       # 动作等级定义
        self.improve_advise=""      # 动作改进建议
        self.left_elbow_angle=0     # 左肘角度
        self.right_elbow_angle=0    # 右肘角度
        self.left_elbow_angle_deviation = 0  # 左肘角度偏差
        self.right_elbow_angle_deviation = 0 # 右肘角度偏差
    
    def _save_image(self, image, filename):
        """在线程中保存图片的函数"""
        cv2.imwrite(filename, image)
        print(f"图片已保存至: {filename}")

    def inference(self, image, show_box=True, show_kpts=True, points=None):
        """执行推理过程，检测边界框和关键点，并计算角度与计数

        参数:
            image (ndarray): 输入图像数组
            show_box (bool): 是否显示检测框
            show_kpts (bool): 是否显示关键点
            points (list of tuples): 参考连线点

        返回:
            tuple: (处理后的图像, 左肘角度, 右肘角度, 计数增量)
        """
        # 预处理图像
        img = PoseUtils.letterbox(image)
        data = PoseUtils.pre_process(img)

        # 模型推理
        pred = self.session.run([self.label_name], {self.input_name: data.astype(np.float32)})[0][0]
        pred = np.transpose(pred, (1, 0))  # [8400, 56]
        pred = pred[pred[:, 4] > self.CONF_THRESHOLD]  # 置信度过滤

        # 未检测到目标时返回默认值
        if len(pred) == 0:
            return image, 0, 0, 0

        # 处理检测结果
        bboxs = self._process_predictions(pred, img.shape, image.shape)

        # 分析并可视化检测结果
        return self._analyze_and_visualize(image, bboxs, show_box, show_kpts, points)


    def _process_predictions(self, pred, input_shape, output_shape):
        """处理模型预测结果"""
        bboxs = PoseUtils.xywh2xyxy(pred)  # 中心宽高转左上右下
        bboxs = PoseUtils.nms(bboxs, iou_thresh=self.IOU_THRESHOLD)  # 非极大值抑制
        bboxs = np.array(PoseUtils.xyxy2xywh(bboxs))  # 左上右下转左上宽高
        return PoseUtils.scale_boxes(input_shape, bboxs, output_shape)  # 坐标缩放到原图

    def _analyze_and_visualize(self, image, bboxs, show_box, show_kpts, points):
        """分析检测结果并可视化"""
        for box in bboxs:
            det_bbox, det_score, kpts = box[0:4], box[4], box[5:]

            # 可视化边界框
            if show_box:
                self._draw_bbox(image, det_bbox, det_score)

            # 可视化关键点和骨架
            if show_kpts:
                PoseUtils.plot_skeleton_kpts(image, kpts)
            
            # 关键点预处理
            kpts_map = PoseUtils.store_keypoints_info(kpts)

            # 过线开启计数统计
            add_count = self._update_count(kpts_map, points)

            # 等级评定
            if add_count==1:    

                # 计算左右肘角度
                self.left_elbow_angle, self.right_elbow_angle = self._calculate_elbow_angles(kpts_map)

                if self.left_elbow_angle <=30 and self.right_elbow_angle <=30:
                    # 优秀
                    self.action_levels = self.ACTION_LEVELS[0]

                elif 30 < self.left_elbow_angle <=60 and 30 < self.right_elbow_angle <=60:
                    # 良好
                    self.action_levels = self.ACTION_LEVELS[1]

                else:
                    # 一般
                    self.action_levels = self.ACTION_LEVELS[2] 

                # 角度偏差
                self.left_elbow_angle_deviation = round(self.left_elbow_angle-30, 2)
                self.right_elbow_angle_deviation = round(self.right_elbow_angle-30, 2)

                # 保存当前帧命名为时间戳,保存路径self.save_image_path
                if self.save_image_path and self.is_save_image:
                    if not os.path.exists(self.save_image_path):
                        os.makedirs(self.save_image_path)

                    # 获取当前时间戳（包含微秒）
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
                    image_filename = os.path.join(self.save_image_path, f"{timestamp}.jpg")

                    # 创建并启动保存线程
                    save_thread = threading.Thread(
                        target=self._save_image,
                        args=(image.copy(), image_filename)  # 使用image.copy()避免线程间的图像数据竞争
                    )
                    save_thread.start()
            
            # 改进建议
            if self.action_levels == self.ACTION_LEVELS[2]:
                self.improve_advise = self.IMPROVE_LIST[1]

            return {
                'image': image,
                'left_elbow_angle': self.left_elbow_angle,
                'right_elbow_angle': self.right_elbow_angle,
                'add_count': add_count,
                'action_levels': self.action_levels,
                'improve_advise': self.improve_advise,
                'left_elbow_angle_deviation': self.left_elbow_angle_deviation,
                'right_elbow_angle_deviation': self.right_elbow_angle_deviation
            }
            # return image, left_elbow_angle, right_elbow_angle, add_count,self.action_levels,self.improve_advise,left_elbow_angle_deviation,right_elbow_angle_deviation

        # 如果 bboxs 为空，返回默认值（理论上不会发生，因已在之前过滤）
        return {
            'image': image,
            'left_elbow_angle': 0,
            'right_elbow_angle': 0,
            'add_count': 0,
            'action_levels': self.action_levels,
            'improve_advise': self.improve_advise,
            'left_elbow_angle_deviation': 0,
            'right_elbow_angle_deviation': 0
        }

    def _draw_bbox(self, image, bbox, score):
        """绘制边界框和置信度"""
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text_pos_y = y1 + 25 if y1 < 30 else y1 - 5
        cv2.putText(image, f"conf:{score:.2f}", (x1 + 5, text_pos_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1)

    def _calculate_elbow_angles(self, kpts_map):
        """计算左右肘角度并保留两位小数"""
        left_angle, right_angle = calculate_elbow_angle(kpts_map)
        return round(left_angle, 2), round(right_angle, 2)

    def _calculate_midpoint(self, left_wrist, right_wrist):
        """计算左右腕中点坐标"""
        if left_wrist is None or right_wrist is None:
            return None
        return ((left_wrist[0] + right_wrist[0]) / 2, (left_wrist[1] + right_wrist[1]) / 2)

    def _update_count(self, kpts_map, points):
        """更新计数逻辑"""
        # 检查关键点是否存在
        if not (3 in kpts_map and 4 in kpts_map):
            return 0

        # 计算中点
        midpoint = self._calculate_midpoint(kpts_map[3], kpts_map[4])
        if midpoint is None or not points or len(points) < 2:
            return 0

        point1, point2 = points[0], points[1]
        # 检查中点是否越过标记线
        is_above = is_midpoint_above_line(midpoint, point1, point2)

        if is_above:
            self.was_above = True
            return 0

        if self.was_above:
            current_time = time.time()
            if current_time - self.last_count_time >= self.COUNT_TIME_THRESHOLD:
                self.last_count_time = current_time
                self.was_above = False
                return 1

        self.was_above = False
        return 0

    def update_save_image_flage(self, state):
        self.is_save_image = state
        # 写入配置
        self.config.set("Settings", "is_save_image", str(state))
        Utils.save_ini_config(self.config, "./config.ini")