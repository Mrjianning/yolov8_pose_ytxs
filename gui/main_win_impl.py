import cv2
import time
from core.pose_detection import Keypoint  # 检测模块
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog,QDialog
from utils.utils import Utils

import matplotlib.pyplot as plt

class PoseDetectionApp:
    def __init__(self, ui):
        self.ui = ui
        self.start_flag = True  # 开始标志位
        self.cap = None 

        self.config=Utils.load_ini_config("./config.ini")
        self.model_path=self.config.get("Model","model_path",fallback='./models/yolo11n-pose.onnx')
        self.video_type=self.config.getint("Model","video_type",fallback=0)                              # 0 视频 1 摄像头                            
        self.is_save_image = self.config.getboolean("Settings", "is_save_image", fallback=False)         # 是否保存图片

        # 安全设置状态（阻塞信号避免意外触发）
        checkbox = self.ui.checkBox_save_image
        checkbox.blockSignals(True)
        checkbox.setChecked(self.is_save_image)
        checkbox.blockSignals(False)
        
        # 连接信号（如果尚未连接）
        if not checkbox.receivers(checkbox.stateChanged):
            checkbox.stateChanged.connect(self.on_save_image_changed)

        self.show_box = True    # 显示检测框
        self.show_kpts = True   # 显示关键点
        self.drawing = False    # 画线标志位
        self.points = []        # 存储选择的点（frame坐标系）
        self.frame = None       # 当前帧
        self.count_num=0        # 计数器

        # 初始化模型
        self.keydet = Keypoint(self.model_path)

        # 绑定鼠标事件到label_video
        self.ui.label_video.mousePressEvent = self.mouse_press_event

    def start_operation(self):
        self.start_flag = True
        print("开始操作")
        self.ui.label_count.clear()
        self.count_num=0
        self.run()

    def stop_operation(self):
        print("停止操作")
        self.start_flag = False
        self.ui.label_video.clear()
        if self.cap is not None:
            self.cap.release()
        print("个数：",self.ui.label_count.text())
        
        cv2.destroyAllWindows()

    def select_video(self):
        print("选择视频")
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            print("选择的文件路径:", file_path)
            self.ui.lineEdit_video_path.setText(file_path)

    def select_model(self, index):
        print("选择模式")
        if index == 0:
            print("选择了视频模式")
            self.video_type = 0
        elif index == 1:
            print("选择了摄像头模式")
            self.video_type = 1
    
    def save_image(self,state):
        self.is_save_image=(state == Qt.Checked)
        print("是否保存图片",self.is_save_image)
        self.keydet.update_save_image_flage(self.is_save_image)

    def show_box_changed(self, state):
        self.show_box = (state == Qt.Checked)

    def show_kpts_changed(self, state):
        self.show_kpts = (state == Qt.Checked)

    def draw_line(self):
        print("绘制连线")
        self.ui.pushButton_draw_line.setEnabled(False)
        self.ui.pushButton_draw_line.setText("绘制中...")
        self.ui.label_video.setCursor(Qt.CrossCursor)
        self.drawing = True
        self.points = []  # 重置点列表

    def map_to_frame(self, x, y):
        """将Qt控件坐标映射到frame实际坐标"""
        if self.frame is None:
            return x, y
        
        # 获取label_video的尺寸
        label_width = self.ui.label_video.width()
        label_height = self.ui.label_video.height()
        
        # 获取frame的实际尺寸
        frame_height, frame_width = self.frame.shape[:2]
        
        # 计算缩放比例
        scale_x = frame_width / label_width
        scale_y = frame_height / label_height
        
        # 映射坐标
        frame_x = int(x * scale_x)
        frame_y = int(y * scale_y)
        
        # 确保坐标在frame范围内
        frame_x = max(0, min(frame_x, frame_width - 1))
        frame_y = max(0, min(frame_y, frame_height - 1))
        
        return frame_x, frame_y

    def mouse_press_event(self, event):
        if self.drawing and event.button() == Qt.LeftButton:
            # 获取鼠标点击位置（相对于label_video）
            x = event.pos().x()
            y = event.pos().y()
            
            # 映射到frame坐标系
            frame_x, frame_y = self.map_to_frame(x, y)
            self.points.append((frame_x, frame_y))
            
            if len(self.points) == 2:
                self.draw_line_on_frame()
                self.finish_drawing()
            else:
                if self.frame is not None:
                    frame_copy = self.frame.copy()
                    cv2.circle(frame_copy, self.points[0], 5, (0, 0, 255), -1)
                    self.display_frame(frame_copy)

    def draw_line_on_frame(self):
        if self.frame is not None and len(self.points) == 2:
            frame_copy = self.frame.copy()
            cv2.line(frame_copy, self.points[0], self.points[1], (0, 255, 0), 2)
            cv2.circle(frame_copy, self.points[0], 5, (0, 0, 255), -1)
            cv2.circle(frame_copy, self.points[1], 5, (0, 0, 255), -1)
            self.display_frame(frame_copy)

    def finish_drawing(self):
        self.drawing = False
        self.ui.pushButton_draw_line.setEnabled(True)
        self.ui.pushButton_draw_line.setText("绘制连线")
        self.ui.label_video.setCursor(Qt.ArrowCursor)

    def display_frame(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.ui.label_video.setPixmap(QPixmap.fromImage(q_img))
    
    def show_charts(self):
        print("显示图表窗口")

        # 4. 在弹窗上显示图表
        # 这里可以添加代码来绘制图表，使用 matplotlib 或其他库
        # 例如：
        plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
        plt.show()


    def run(self):
        
        # 视频源
        if self.video_type == 0:
            videopath = self.ui.lineEdit_video_path.text()
            print(videopath)
        elif self.video_type == 1:
            videopath = 0
        else:
            print("\033[1;91m 输入错误，请检查mode的赋值 \033[0m")
            return

        # 打开视频源
        self.cap = cv2.VideoCapture(videopath)
        if not self.cap.isOpened():
            print("\033[1;91m 无法打开视频源，请检查路径或设备 \033[0m")
            return

        # 初始化FPS计数器
        start_time = time.time()
        counter = 0

        try:
            while self.start_flag:
                ret, frame = self.cap.read()
                if not ret:
                    print("\033[1;91m 无法读取到下一帧 \033[0m")
                    break

                # 检测
                result=self.keydet.inference(frame, self.show_box, self.show_kpts,self.points)
                # 获取检测结果
                image=result["image"]
                left_elbow_angle=result["left_elbow_angle"]
                right_elbow_angle=result["right_elbow_angle"]
                add_count=result["add_count"]
                action_level=result["action_levels"]
                improve_advise=result["improve_advise"]
                left_elbow_angle_deviation=result["left_elbow_angle_deviation"]
                right_elbow_angle_deviation=result["right_elbow_angle_deviation"]
    
                # 保存当前帧
                self.frame = image  
                self.count_num+=add_count

                # 显示角度到界面
                self.ui.label_left_angle.setText(str(left_elbow_angle))
                self.ui.label_right_angle.setText(str(right_elbow_angle))

                # 显示次数到界面
                self.ui.label_count.setText(str(self.count_num))

                # 显示动作等级到界面
                self.ui.label_action_level.setText(action_level)

                # 显示改进建议到界面
                self.ui.label_improve_advise.setText(improve_advise)

                # 显示角度偏差到界面
                self.ui.label_left_angle_deviation.setText(str(left_elbow_angle_deviation))
                self.ui.label_right_angle_deviation.setText(str(right_elbow_angle_deviation))

                # 处理绘制线
                if self.drawing and len(self.points) == 1:
                    cv2.circle(image, self.points[0], 5, (0, 0, 255), -1)

                elif len(self.points) == 2:
                    cv2.line(image, self.points[0], self.points[1], (0, 255, 0), 2)
                    cv2.circle(image, self.points[0], 5, (0, 0, 255), -1)
                    cv2.circle(image, self.points[1], 5, (0, 0, 255), -1)

                # 计算并显示FPS
                counter += 1
                if (time.time() - start_time) != 0:
                    fps = float('%.1f' % (counter / (time.time() - start_time)))
                    cv2.putText(image, f"FPS:{fps}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
                    self.display_frame(image)

                cv2.waitKey(1)

        except Exception as e:
            print("\033[1;91m 发生异常：", e, "\033[0m")

        finally:
            self.cap.release()
            print("视频捕获资源已释放")