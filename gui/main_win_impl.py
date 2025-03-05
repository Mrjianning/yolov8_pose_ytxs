import cv2
import time
from core.pose_detection import Keypoint  # 检测模块
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog


class PoseDetectionApp:
    def __init__(self, ui):
        self.ui = ui
        self.start_flag = True  # 开始标志位
        self.cap = None         
        self.model = 0          # 0 视频 1 摄像头
        self.show_box = True    # 显示检测框
        self.show_kpts = True   # 显示关键点
        self.drawing = False    # 画线标志位
        self.points = []        # 存储选择的点（frame坐标系）
        self.frame = None       # 当前帧
        self.count_num=0        # 计数器
        
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
            self.model = 0
        elif index == 1:
            print("选择了摄像头模式")
            self.model = 1
    
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

    def run(self):
        modelpath = r'models/yolo11n-pose.onnx'
        keydet = Keypoint(modelpath)
        
        mode = self.model
        if mode == 0:
            videopath = self.ui.lineEdit_video_path.text()
            print(videopath)
        elif mode == 1:
            videopath = 0
        else:
            print("\033[1;91m 输入错误，请检查mode的赋值 \033[0m")
            return

        self.cap = cv2.VideoCapture(videopath)
        if not self.cap.isOpened():
            print("\033[1;91m 无法打开视频源，请检查路径或设备 \033[0m")
            return

        start_time = time.time()
        counter = 0

        try:
            while self.start_flag:
                ret, frame = self.cap.read()
                if not ret:
                    print("\033[1;91m 无法读取到下一帧 \033[0m")
                    break

                # 检测
                image, left_elbow_angle, right_elbow_angle,add_count = keydet.inference(frame, self.show_box, self.show_kpts,self.points)
                self.frame = image  # 保存当前帧
                self.count_num+=add_count

                # 显示角度到界面
                self.ui.label_left_angle.setText(str(left_elbow_angle))
                self.ui.label_right_angle.setText(str(right_elbow_angle))

                # 显示次数到界面
                self.ui.label_count.setText(str(self.count_num))

                # 处理绘制
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