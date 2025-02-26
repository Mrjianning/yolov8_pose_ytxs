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
        self.model=0            # 0 视频 1 摄像头
        self.show_box=True      # 显示检测框
        self.show_kpts=True     # 显示关键点

    def start_operation(self):
        self.start_flag = True
        print("开始操作")
        self.run()

    def stop_operation(self):
        print("停止操作")
        self.start_flag = False
        self.ui.label_video.clear()
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def select_video(self):
        print("选择视频")
        # 打开文件对话框，选择视频文件
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi)")

        # 打印文件路径
        if file_dialog.exec_():  # 执行对话框
            file_path = file_dialog.selectedFiles()[0]  # 获取选择的文件路径
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
    
    def show_box_changed(self,state):
       if state == Qt.Checked:
            self.show_box = True
       else:
            self.show_box = False

    def show_kpts_changed(self,state):
        if state == Qt.Checked:
            self.show_kpts = True
        else:
            self.show_kpts = False
            
    def run(self):
        # 模型路径
        modelpath = r'models/yolo11n-pose.onnx'
        # 实例化模型
        keydet = Keypoint(modelpath)
        
        mode = self.model

        # 0 视频 1 摄像头
        if mode == 0:
            # 获取视频路径
            videopath = self.ui.lineEdit_video_path.text()
            print(videopath)
            
        elif mode == 1:
            videopath = 0

        else:
            print("\033[1;91m 输入错误，请检查mode的赋值 \033[0m")
            return

        # 打开视频
        self.cap = cv2.VideoCapture(videopath)
        if not self.cap.isOpened():
            print("\033[1;91m 无法打开视频源，请检查路径或设备 \033[0m")
            return

        # 返回当前时间
        start_time = time.time()
        counter = 0

        try:
            while self.start_flag:
                # 从摄像头中读取一帧图像
                ret, frame = self.cap.read()
                if not ret:
                    print("\033[1;91m 无法读取到下一帧 \033[0m")
                    break

                # 检测
                image,left_elbow_angle,right_elbow_angle = keydet.inference(frame, self.show_box, self.show_kpts)

                # 显示到界面
                self.ui.label_left_angle.setText(str(left_elbow_angle))
                self.ui.label_right_angle.setText(str(right_elbow_angle))

                counter += 1  # 计算帧数
                # 实时显示帧数
                if (time.time() - start_time) != 0:
                    cv2.putText(image, "FPS:{0}".format(float('%.1f' % (counter / (time.time() - start_time)))), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)

                    # 将OpenCV图像转换为QImage
                    height, width,chanel= image.shape
                    bytesPerLine = 3 * width
                    qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                    
                    # 显示图像
                    self.ui.label_video.setPixmap(QPixmap.fromImage(qImg))
                cv2.waitKey(1)

        except Exception as e:
            print("\033[1;91m 发生异常：", e, "\033[0m")

        finally:
            # 释放视频捕获资源
            self.cap.release()
            print("视频捕获资源已释放")
