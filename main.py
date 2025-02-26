import sys

from gui.main_win import Ui_MainWindow   # GUI模块
from PyQt5.QtWidgets import QMainWindow, QApplication
from gui.main_win_impl import PoseDetectionApp


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        self._ui=Ui_MainWindow()
        
        self._ui.setupUi(self)

        # 初始化信号和槽
        self.connect_signals_slots()

        # 初始化PoseDetectionApp
        self._pose_detection_app = PoseDetectionApp(self._ui)

    # 连接信号和槽
    def connect_signals_slots(self):

        # 连接信号和槽
        self._ui.button_start.clicked.connect(self.on_button_start_clicked_)        # 开始按钮
        self._ui.button_stop.clicked.connect(self.on_button_stop_clicked_)          # 停止按钮
        self._ui.button_select_video.clicked.connect(self.on_button_select_video_clicked_)  # 选择视频按钮
        self._ui.comboBox_select_model.currentIndexChanged.connect(self.on_comboBox_select_model_currentIndexChanged_)  # 选择模型下拉框
        self._ui.checkBox_show_box.stateChanged.connect(self.on_checkBox_show_box_stateChanged_)                        # 显示检测框复选框
        self._ui.checkBox_show_kpts.stateChanged.connect(self.on_checkBox_show_kpts_stateChanged_)                      # 显示关键点复选框
        
    # 开始按钮
    def on_button_start_clicked_(self):
        self._pose_detection_app.start_operation()

    # 停止按钮
    def on_button_stop_clicked_(self):
        self._pose_detection_app.stop_operation()

    # 选择视频按钮
    def on_button_select_video_clicked_(self):
        self._pose_detection_app.select_video()

    # 选择模型下拉框
    def on_comboBox_select_model_currentIndexChanged_(self, index):
        self._pose_detection_app.select_model(index)

    # 显示检测框复选框
    def on_checkBox_show_box_stateChanged_(self, state):
        self._pose_detection_app.show_box_changed(state)

    # 显示关键点复选框
    def on_checkBox_show_kpts_stateChanged_(self, state):
        self._pose_detection_app.show_kpts_changed(state)

if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())  # 进入主循环
