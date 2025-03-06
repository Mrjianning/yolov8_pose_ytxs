import onnxruntime
import numpy as np
import cv2
import time
from core.pose_calculator import calculate_elbow_angle,is_midpoint_above_line

# 姿态检测模块

# 调色板
palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])
# 17个关键点连接顺序
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
# 骨架颜色
pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
# 关键点颜色
pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    '''  调整图像大小和两边灰条填充  '''
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # 缩放比例 (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # 只进行下采样 因为上采样会让图片模糊
    if not scaleup:
        r = min(r, 1.0)
    # 计算pad长宽
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 保证缩放后图像比例不变
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # 在较小边的两侧进行pad, 而不是在一侧pad
    dw /= 2
    dh /= 2
    # 将原图resize到new_unpad（长边相同，比例相同的新图）
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    # 计算上下两侧的padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    # 计算左右两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # 添加灰条
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im

def pre_process(img):
    # 归一化 调整通道为（1，3，640，640）
    img = img / 255.
    img = np.transpose(img, (2, 0, 1))
    data = np.expand_dims(img, axis=0)
    return data

def xywh2xyxy(x):
    ''' 中心坐标、w、h ------>>> 左上点，右下点 '''
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

# nms算法
def nms(dets, iou_thresh):
    # dets: N * M, N是bbox的个数，M的前4位是对应的 左上点，右下点
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 求每个bbox的面积
    order = scores.argsort()[::-1]  # 对分数进行倒排序
    keep = []  # 用来保存最后留下来的bboxx下标
    while order.size > 0:
        i = order[0]  # 无条件保留每次迭代中置信度最高的bbox
        keep.append(i)
        # 计算置信度最高的bbox和其他剩下bbox之间的交叉区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算置信度高的bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的必烈
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留ovr小于thresh的bbox，进入下一次迭代。
        inds = np.where(ovr <= iou_thresh)[0]
        # 因为ovr中的索引不包括order[0]所以要向后移动一位
        order = order[inds + 1]
    output = []
    for i in keep:
        output.append(dets[i].tolist())
    return np.array(output)

def xyxy2xywh(a):
    ''' 左上点 右下点 ------>>> 左上点 宽 高 '''
    b = np.copy(a)
    # y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    # y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    b[:, 2] = a[:, 2] - a[:, 0]  # w
    b[:, 3] = a[:, 3] - a[:, 1]  # h
    return b

def scale_boxes(img1_shape, boxes, img0_shape):
    '''   将预测的坐标信息转换回原图尺度
    :param img1_shape: 缩放后的图像尺度
    :param boxes:  预测的box信息
    :param img0_shape: 原始图像尺度
    '''
    # 将检测框(x y w h)从img1_shape(预测图) 缩放到 img0_shape(原图)
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    boxes[:, 0] -= pad[0]
    boxes[:, 1] -= pad[1]
    boxes[:, :4] /= gain  # 检测框坐标点还原到原图上
    num_kpts = boxes.shape[1] // 3   # 56 // 3 = 18
    for kid in range(2,num_kpts+1):
        boxes[:, kid * 3-1] = (boxes[:, kid * 3-1] - pad[0]) / gain
        boxes[:, kid * 3 ]  = (boxes[:, kid * 3 ] -  pad[1]) / gain
    # boxes[:, 5:] /= gain  # 关键点坐标还原到原图上
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    # 进行一个边界截断，以免溢出
    # 并且将检测框的坐标（左上角x，左上角y，宽度，高度）--->>>（左上角x，左上角y，右下角x，右下角y）
    top_left_x = boxes[:, 0].clip(0, shape[1])
    top_left_y = boxes[:, 1].clip(0, shape[0])
    bottom_right_x = (boxes[:, 0] + boxes[:, 2]).clip(0, shape[1])
    bottom_right_y = (boxes[:, 1] + boxes[:, 3]).clip(0, shape[0])
    boxes[:, 0] = top_left_x      #左上
    boxes[:, 1] = top_left_y
    boxes[:, 2] = bottom_right_x  #右下
    boxes[:, 3] = bottom_right_y

# 存储关键点信息
def store_keypoints_info(kpts, steps=3):
    num_kpts = len(kpts) // steps  # 51 / 3 = 17
    keypoints_info = {}

    for kid in range(num_kpts):
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        conf = kpts[steps * kid + 2]
        if conf > 0.5:  # 只存储置信度大于 0.5 的关键点
            keypoints_info[kid] = (int(x_coord), int(y_coord), conf)

    return keypoints_info

def plot_skeleton_kpts(im, kpts, steps=3):
    num_kpts = len(kpts) // steps  # 51 / 3 =17
    # 画点
    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        conf = kpts[steps * kid + 2]
        if conf > 0.5:   # 关键点的置信度必须大于 0.5
            cv2.circle(im, (int(x_coord), int(y_coord)), 5, (int(r), int(g), int(b)), -1)

            # 在关键点旁边标记序号
            cv2.putText(im, str(kid), (int(x_coord), int(y_coord)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 画骨架
    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        conf1 = kpts[(sk[0]-1)*steps+2]
        conf2 = kpts[(sk[1]-1)*steps+2]
        if conf1 >0.5 and conf2 >0.5:  # 对于肢体，相连的两个关键点置信度 必须同时大于 0.5
            cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

class Keypoint:
    """基于 ONNX 模型的关键点检测类"""
    ACTION_LEVELS = ["优秀", "良好", "一般"]  # 动作等级定义
    CONF_THRESHOLD = 0.7  # 置信度阈值
    IOU_THRESHOLD = 0.6   # NMS 的 IoU 阈值
    COUNT_TIME_THRESHOLD = 1.0  # 计数时间间隔阈值（秒）

    def __init__(self, modelpath):
        """初始化 Keypoint 检测器"""
        print("==========初始化==========")
        self.session = onnxruntime.InferenceSession(modelpath, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.label_name = self.session.get_outputs()[0].name
        self.last_count_time = 0.0  # 上次计数时间戳
        self.was_above = False      # 中点是否在上方的状态

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
        img = letterbox(image)
        data = pre_process(img)

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
        bboxs = xywh2xyxy(pred)  # 中心宽高转左上右下
        bboxs = nms(bboxs, iou_thresh=self.IOU_THRESHOLD)  # 非极大值抑制
        bboxs = np.array(xyxy2xywh(bboxs))  # 左上右下转左上宽高
        return scale_boxes(input_shape, bboxs, output_shape)  # 坐标缩放到原图

    def _analyze_and_visualize(self, image, bboxs, show_box, show_kpts, points):
        """分析检测结果并可视化"""
        for box in bboxs:
            det_bbox, det_score, kpts = box[0:4], box[4], box[5:]

            # 可视化边界框
            if show_box:
                self._draw_bbox(image, det_bbox, det_score)

            # 可视化关键点和骨架
            if show_kpts:
                plot_skeleton_kpts(image, kpts)

            # 关键点预处理
            kpts_map = store_keypoints_info(kpts)
            # 计算
            left_elbow_angle, right_elbow_angle = self._calculate_elbow_angles(kpts_map)
            # 计数统计
            add_count = self._update_count(kpts_map, points)

            return image, left_elbow_angle, right_elbow_angle, add_count

        # 如果 bboxs 为空，返回默认值（理论上不会发生，因已在之前过滤）
        return image, 0, 0, 0

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