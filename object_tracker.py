import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QHBoxLayout, QFileDialog, QLabel, QWidget, QComboBox, 
                            QMessageBox, QSlider, QFrame, QSplitter, QGroupBox, QGridLayout)
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QPen, QColor, QFont, QIcon, 
                        QPalette, QBrush, QLinearGradient, QCursor)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect, QPoint, QSize, QTimer
import os
# 确保中文显示正常
import matplotlib
matplotlib.use('Agg')

from components.PromptVT.lib.test.tracker.PromptVT import PromptVT
from components.PromptVT.lib.test.parameter.PromptVT import parameters

try:
    from components.PromptVT.lib.test.evaluation import Tracker
except ImportError:
    print("pytracking模块未找到，请确保已正确安装pytracking")
    # 为了代码能够运行，创建一个模拟的Tracker类
    class Tracker:
        def __init__(self, name, parameter_name):
            self.name = name
            self.parameter_name = parameter_name
            print(f"模拟Tracker初始化: {name}, {parameter_name}")
        
        def initialize(self, image, box):
            print(f"模拟Tracker初始化，目标框: {box}")
            self.state = box
            return {"state": box}
        
        def track(self, image):
            # 返回一个随机移动的框作为模拟
            h, w = image.shape[:2]
            x, y, w_box, h_box = self.state
            x += np.random.randint(-5, 6)
            y += np.random.randint(-5, 6)
            x = max(0, min(x, w - w_box))
            y = max(0, min(y, h - h_box))
            self.state = (x, y, w_box, h_box)
            return {"state": self.state}

def _build_init_info(box):
    return {'init_bbox': box}

class TrackingThread(QThread):
    """跟踪线程，负责在后台运行目标跟踪算法"""
    update_frame = pyqtSignal(np.ndarray)
    tracking_finished = pyqtSignal()
    
    def __init__(self, video_capture, tracker, init_box=None):
        super().__init__()
        self.video_capture = video_capture
        self.tracker = tracker
        self.init_box = init_box  # 初始目标框 (x, y, w, h)
        self.running = False
        self.paused = False
        self.current_frame = None
        self.intervals = 20
        self.conf = 0.5
        self.search_factor = 5
        
    def run(self):
        """线程运行函数"""
        self.running = True
        frame_count = 0
        
        while self.running:
            if self.paused:
                self.msleep(100)
                continue
                
            ret, frame = self.video_capture.read()
            if not ret:
                break
                
            self.current_frame = frame
            
            if frame_count == 0 and self.init_box:
                # 初始化跟踪器
                self.tracker.initialize(frame, _build_init_info(self.init_box))
            elif frame_count > 0:
                # 跟踪目标
                output = self.tracker.track(frame,None,self.intervals, self.search_factor, self.conf)
                state = output['target_bbox']  # 跟踪结果框 (x, y, w, h)
                
                # 在帧上绘制跟踪框
                x, y, w, h = [int(s) for s in state]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Tracking", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 发送更新后的帧
            self.update_frame.emit(frame)
            frame_count += 1
            
            # 控制帧率
            self.msleep(30)
            
        self.tracking_finished.emit()
        
    def stop(self):
        """停止跟踪线程"""
        self.running = False
        self.wait()
        
    def pause(self):
        """暂停跟踪"""
        self.paused = True
        
    def resume(self):
        """恢复跟踪"""
        self.paused = False

class ObjectTrackerApp(QMainWindow):
    """主窗口类"""
    def __init__(self):
        super().__init__()
        
        self.tracker = None
        self.tracking_thread = None
        self.video_capture = None
        self.init_box = None
        self.selecting_region = False
        self.region_selected = False
        self.drawing_rect = QRect()
        self.current_frame = None
        self.tracker_name = "dimp"
        self.tracker_param = "dimp50"
        self.drag_point = None
        self.resize_direction = None
        self.resizing = False
        
        # 自定义样式设置
        self.setup_styles()
        
        self.init_ui()
        
    def setup_styles(self):
        """设置自定义样式"""
        # 主色调
        self.primary_color = "#2D5BFF"
        self.secondary_color = "#00C853"
        self.tertiary_color = "#FF6D00"
        self.dark_color = "#212121"
        self.light_color = "#F5F5F5"
        
        # 样式表
        self.stylesheet = f"""
            QMainWindow {{
                background-color: {self.light_color};
                color: {self.dark_color};
            }}
            
            QLabel {{
                color: {self.dark_color};
                font-size: 14px;
            }}
            
            QPushButton {{
                background-color: {self.primary_color};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: 500;
            }}
            
            QPushButton:hover {{
                background-color: #1E40AF;
            }}
            
            QPushButton:pressed {{
                background-color: #0D2259;
            }}
            
            QPushButton:disabled {{
                background-color: #BDBDBD;
                color: #757575;
            }}
            
            QComboBox {{
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                padding: 6px;
                background-color: white;
                font-size: 14px;
            }}
            
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 1px;
                border-left-color: #BDBDBD;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }}
            
            QSlider::groove:horizontal {{
                border: 1px solid #BDBDBD;
                height: 8px;
                background: white;
                margin: 2px 0;
                border-radius: 4px;
            }}
            
            QSlider::handle:horizontal {{
                background: {self.primary_color};
                border: 1px solid #5C5C5C;
                width: 16px;
                margin: -2px 0;
                border-radius: 4px;
            }}
            
            QGroupBox {{
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                margin-top: 10px;
                font-size: 14px;
                font-weight: bold;
            }}
            
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                background-color: {self.light_color};
            }}
            
            #status_label {{
                font-size: 16px;
                font-weight: bold;
                color: {self.primary_color};
                padding: 8px;
                border-radius: 4px;
                background-color: rgba(45, 91, 255, 0.1);
            }}
            
            #video_label {{
                border: 2px dashed #BDBDBD;
                border-radius: 8px;
                background-color: white;
            }}
        """
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('高级单目标跟踪应用')
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(self.stylesheet)
        
        # 创建中央部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # 创建顶部标题区域
        title_layout = QHBoxLayout()
        title_label = QLabel('高级单目标跟踪应用')
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet(f"color: {self.primary_color};")
        
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        main_layout.addLayout(title_layout)
        
        # 创建视频显示和控制面板区域
        content_splitter = QSplitter(Qt.Vertical)
        
        # 视频显示区域
        video_group = QGroupBox("视频预览")
        video_layout = QVBoxLayout(video_group)
        
        self.video_label = QLabel('请选择视频源并开始跟踪')
        self.video_label.setObjectName("video_label")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #f0f0f0;")
        self.video_label.mousePressEvent = self.mouse_press_event
        self.video_label.mouseMoveEvent = self.mouse_move_event
        self.video_label.mouseReleaseEvent = self.mouse_release_event
        
        video_layout.addWidget(self.video_label)
        
        content_splitter.addWidget(video_group)
        
        # 控制面板区域
        control_group = QGroupBox("控制面板")
        control_layout = QVBoxLayout(control_group)
        
        # 视频源选择
        source_frame = QFrame()
        source_frame.setFrameShape(QFrame.StyledPanel)
        source_layout = QHBoxLayout(source_frame)
        
        source_label = QLabel('视频源:')
        self.source_combo = QComboBox()
        self.source_combo.addItems(['摄像头', '视频文件'])
        source_layout.addWidget(source_label)
        source_layout.addWidget(self.source_combo)        
        
        # 打开按钮                                    
        self.open_button = QPushButton('打开')
        self.open_button.setIcon(QIcon.fromTheme("document-open"))
        self.open_button.clicked.connect(self.open_source)
        source_layout.addWidget(self.open_button)
        
        source_layout.addStretch()
        
        control_layout.addWidget(source_frame)
        
        # 跟踪控制按钮
        control_frame = QFrame()
        control_frame.setFrameShape(QFrame.StyledPanel)
        control_buttons_layout = QHBoxLayout(control_frame)
        
        self.start_button = QPushButton('开始跟踪')
        self.start_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_tracking)
        
        self.stop_button = QPushButton('停止跟踪')
        self.stop_button.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_tracking)
        
        self.reset_button = QPushButton('重新选择目标')
        self.reset_button.setIcon(QIcon.fromTheme("view-refresh"))
        self.reset_button.setEnabled(False)
        self.reset_button.clicked.connect(self.reset_tracking)
        
        control_buttons_layout.addWidget(self.start_button)
        control_buttons_layout.addWidget(self.stop_button)
        control_buttons_layout.addWidget(self.reset_button)
        
        control_buttons_layout.addStretch()
        
        control_layout.addWidget(control_frame)
        
        # 跟踪器选择
        tracker_frame = QFrame()
        tracker_frame.setFrameShape(QFrame.StyledPanel)
        tracker_layout = QHBoxLayout(tracker_frame)
        
        tracker_label = QLabel('跟踪器:')
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(['pvt'])
        tracker_layout.addWidget(tracker_label)
        tracker_layout.addWidget(self.tracker_combo)
        
        tracker_info_label = QLabel('当前跟踪器: pvt')
        tracker_info_label.setStyleSheet("color: #757575; font-size: 12px;")
        tracker_layout.addWidget(tracker_info_label)
        
        control_layout.addWidget(tracker_frame)
        
        # 状态标签
        self.status_label = QLabel('状态: 就绪')
        self.status_label.setObjectName("status_label")
        control_layout.addWidget(self.status_label)
        
        content_splitter.addWidget(control_group)
        
        # 设置分割器比例
        content_splitter.setSizes([600, 200])
        
        main_layout.addWidget(content_splitter)
        
        # 底部信息
        footer_layout = QHBoxLayout()
        footer_label = QLabel('单目标跟踪应用 V1.0 | 使用PyQt和pytracking开发')
        footer_label.setStyleSheet("color: #757575; font-size: 12px;")
        footer_layout.addWidget(footer_label)
        footer_layout.addStretch()
        
        main_layout.addLayout(footer_layout)
        
    def open_source(self):
        """打开视频源"""
        source_type = self.source_combo.currentText()
        
        if self.tracking_thread and self.tracking_thread.isRunning():
            self.tracking_thread.stop()
            
        if self.video_capture:
            self.video_capture.release()
            
        if source_type == '摄像头':
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                self.status_label.setText('状态: 无法打开摄像头')
                return
            self.status_label.setText('状态: 已连接摄像头')
        else:  # 视频文件
            file_path, _ = QFileDialog.getOpenFileName(
                self, '打开视频文件', '', '视频文件 (*.mp4 *.avi *.mov *.mkv)')
            
            if not file_path:
                return
                
            self.video_capture = cv2.VideoCapture(file_path)
            if not self.video_capture.isOpened():
                self.status_label.setText('状态: 无法打开视频文件')
                return
            self.status_label.setText(f'状态: 已加载视频文件: {os.path.basename(file_path)}')
        
        # 读取第一帧并显示
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame = frame
            self.display_frame(frame)
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.reset_button.setEnabled(False)
        else:
            self.status_label.setText('状态: 无法读取视频帧')
    
    def start_tracking(self):
        """开始跟踪"""
        if not self.video_capture or not self.init_box:
            self.status_label.setText('状态: 请先选择目标区域')
            return
            
        # 创建跟踪器
        self.create_tracker()
        
        # 重置视频捕获到开始位置
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # 创建并启动跟踪线程
        self.tracking_thread = TrackingThread(self.video_capture, self.tracker, self.init_box)
        self.tracking_thread.update_frame.connect(self.display_frame)
        self.tracking_thread.tracking_finished.connect(self.tracking_finished)
        self.tracking_thread.start()
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.status_label.setText('状态: 正在跟踪...')
    
    def stop_tracking(self):
        """停止跟踪"""
        if self.tracking_thread and self.tracking_thread.isRunning():
            self.tracking_thread.stop()
            self.status_label.setText('状态: 跟踪已停止')
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.reset_button.setEnabled(False)
    
    def reset_tracking(self):
        """重置跟踪，重新选择目标"""
        self.stop_tracking()
        self.init_box = None
        self.region_selected = False
        self.status_label.setText('状态: 请选择新的目标区域')
        
        # 显示第一帧以便重新选择
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame = frame
            self.display_frame(frame)
    
    def tracking_finished(self):
        """跟踪完成回调"""
        self.status_label.setText('状态: 跟踪完成')
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reset_button.setEnabled(False)
    
    def mouse_press_event(self, event):
        """鼠标按下事件"""
        if not self.video_capture or self.tracking_thread and self.tracking_thread.isRunning():
            return
            
        pos = event.pos()
        
        # 检查是否点击了已存在的矩形边界或角落
        if self.region_selected and self.drawing_rect.contains(pos):
            # 检查是否点击了角落或边界
            corner_size = 10
            
            # 检查角落
            top_left = self.drawing_rect.topLeft()
            top_right = self.drawing_rect.topRight()
            bottom_left = self.drawing_rect.bottomLeft()
            bottom_right = self.drawing_rect.bottomRight()
            
            if (pos - top_left).manhattanLength() < corner_size:
                self.resize_direction = 'top_left'
                self.resizing = True
            elif (pos - top_right).manhattanLength() < corner_size:
                self.resize_direction = 'top_right'
                self.resizing = True
            elif (pos - bottom_left).manhattanLength() < corner_size:
                self.resize_direction = 'bottom_left'
                self.resizing = True
            elif (pos - bottom_right).manhattanLength() < corner_size:
                self.resize_direction = 'bottom_right'
                self.resizing = True
            # 检查边界
            elif abs(pos.y() - self.drawing_rect.top()) < corner_size and \
                 self.drawing_rect.left() <= pos.x() <= self.drawing_rect.right():
                self.resize_direction = 'top'
                self.resizing = True
            elif abs(pos.y() - self.drawing_rect.bottom()) < corner_size and \
                 self.drawing_rect.left() <= pos.x() <= self.drawing_rect.right():
                self.resize_direction = 'bottom'
                self.resizing = True
            elif abs(pos.x() - self.drawing_rect.left()) < corner_size and \
                 self.drawing_rect.top() <= pos.y() <= self.drawing_rect.bottom():
                self.resize_direction = 'left'
                self.resizing = True
            elif abs(pos.x() - self.drawing_rect.right()) < corner_size and \
                 self.drawing_rect.top() <= pos.y() <= self.drawing_rect.bottom():
                self.resize_direction = 'right'
                self.resizing = True
            else:
                # 点击了矩形内部，移动矩形
                self.drag_point = pos
                
            self.selecting_region = False
        else:
            # 开始新的选择
            self.selecting_region = True
            self.resizing = False
            self.drawing_rect.setTopLeft(pos)
            self.drawing_rect.setBottomRight(pos)
        
        self.update()
    
    def mouse_move_event(self, event):
        """鼠标移动事件"""
        if not self.video_capture or self.tracking_thread and self.tracking_thread.isRunning():
            return
            
        pos = event.pos()
        
        # 调整鼠标光标形状
        if self.region_selected and not self.selecting_region and not self.resizing:
            corner_size = 10
            
            # 检查角落
            top_left = self.drawing_rect.topLeft()
            top_right = self.drawing_rect.topRight()
            bottom_left = self.drawing_rect.bottomLeft()
            bottom_right = self.drawing_rect.bottomRight()
            
            if (pos - top_left).manhattanLength() < corner_size or \
               (pos - bottom_right).manhattanLength() < corner_size:
                self.video_label.setCursor(Qt.SizeFDiagCursor)
            elif (pos - top_right).manhattanLength() < corner_size or \
                 (pos - bottom_left).manhattanLength() < corner_size:
                self.video_label.setCursor(Qt.SizeBDiagCursor)
            # 检查边界
            elif abs(pos.y() - self.drawing_rect.top()) < corner_size and \
                 self.drawing_rect.left() <= pos.x() <= self.drawing_rect.right():
                self.video_label.setCursor(Qt.SizeVerCursor)
            elif abs(pos.y() - self.drawing_rect.bottom()) < corner_size and \
                 self.drawing_rect.left() <= pos.x() <= self.drawing_rect.right():
                self.video_label.setCursor(Qt.SizeVerCursor)
            elif abs(pos.x() - self.drawing_rect.left()) < corner_size and \
                 self.drawing_rect.top() <= pos.y() <= self.drawing_rect.bottom():
                self.video_label.setCursor(Qt.SizeHorCursor)
            elif abs(pos.x() - self.drawing_rect.right()) < corner_size and \
                 self.drawing_rect.top() <= pos.y() <= self.drawing_rect.bottom():
                self.video_label.setCursor(Qt.SizeHorCursor)
            elif self.drawing_rect.contains(pos):
                self.video_label.setCursor(Qt.OpenHandCursor)
            else:
                self.video_label.setCursor(Qt.ArrowCursor)
        
        # 处理选择或调整大小
        if self.selecting_region:
            # 正在创建新的矩形
            self.drawing_rect.setBottomRight(pos)
            self.display_frame(self.current_frame, self.drawing_rect)
        elif self.resizing:
            # 正在调整现有矩形的大小
            self.resize_rect(pos)
            self.display_frame(self.current_frame, self.drawing_rect)
        elif self.drag_point:
            # 正在移动现有矩形
            dx = pos.x() - self.drag_point.x()
            dy = pos.y() - self.drag_point.y()
            self.drawing_rect.translate(dx, dy)
            self.drag_point = pos
            self.display_frame(self.current_frame, self.drawing_rect)
    
    def mouse_release_event(self, event):
        """鼠标释放事件"""
        if not self.video_capture or self.tracking_thread and self.tracking_thread.isRunning():
            return
            
        pos = event.pos()
        
        if self.selecting_region:
            self.selecting_region = False
            self.drawing_rect.setBottomRight(pos)
            
            # 确保矩形有合理的大小
            if self.drawing_rect.width() > 10 and self.drawing_rect.height() > 10:
                self.region_selected = True
                self.update_init_box()
                self.status_label.setText(f'状态: 已选择目标区域 - 大小: {self.drawing_rect.width()}x{self.drawing_rect.height()}')
            else:
                self.region_selected = False
                self.status_label.setText('状态: 选择的区域太小，请重新选择')
                self.drawing_rect = QRect()
                self.display_frame(self.current_frame)
        elif self.resizing:
            self.resizing = False
            self.resize_direction = None
            self.update_init_box()
            self.status_label.setText(f'状态: 已调整目标区域 - 大小: {self.drawing_rect.width()}x{self.drawing_rect.height()}')
        elif self.drag_point:
            self.drag_point = None
            self.update_init_box()
            self.status_label.setText(f'状态: 已移动目标区域 - 位置: ({self.drawing_rect.left()}, {self.drawing_rect.top()})')
    
    def resize_rect(self, pos):
        """根据方向调整矩形大小"""
        current_rect = QRect(self.drawing_rect)
        
        if self.resize_direction == 'top_left':
            current_rect.setTopLeft(pos)
        elif self.resize_direction == 'top_right':
            current_rect.setTopRight(pos)
        elif self.resize_direction == 'bottom_left':
            current_rect.setBottomLeft(pos)
        elif self.resize_direction == 'bottom_right':
            current_rect.setBottomRight(pos)
        elif self.resize_direction == 'top':
            current_rect.setTop(pos.y())
        elif self.resize_direction == 'bottom':
            current_rect.setBottom(pos.y())
        elif self.resize_direction == 'left':
            current_rect.setLeft(pos.x())
        elif self.resize_direction == 'right':
            current_rect.setRight(pos.x())
        
        # 确保矩形有合理的大小
        if current_rect.width() > 10 and current_rect.height() > 10:
            self.drawing_rect = current_rect
    
    def update_init_box(self):
        """更新初始化框坐标"""
        # 计算在原始图像上的坐标
        img_width, img_height = self.current_frame.shape[1], self.current_frame.shape[0]
        label_width, label_height = self.video_label.width(), self.video_label.height()
        
        # 计算缩放比例
        scale_x = img_width / label_width
        scale_y = img_height / label_height
        
        # 转换为OpenCV格式的矩形 (x, y, w, h)
        x = int(self.drawing_rect.left() * scale_x)
        y = int(self.drawing_rect.top() * scale_y)
        w = int(self.drawing_rect.width() * scale_x)
        h = int(self.drawing_rect.height() * scale_y)
        
        self.init_box = [x, y, w, h]
    
    def display_frame(self, frame, rect=None):
        """显示帧，可选择绘制矩形"""
        # 复制帧以避免修改原始帧
        display_frame = frame.copy()
        
        # 绘制选择的矩形
        if rect:
            x1, y1 = rect.topLeft().x(), rect.topLeft().y()
            x2, y2 = rect.bottomRight().x(), rect.bottomRight().y()
            
            # 确保坐标在有效范围内
            img_width, img_height = display_frame.shape[1], display_frame.shape[0]
            label_width, label_height = self.video_label.width(), self.video_label.height()
            
            # 计算缩放比例
            scale_x = img_width / label_width
            scale_y = img_height / label_height
            
            # 转换为图像坐标
            x1_img = int(x1 * scale_x)
            y1_img = int(y1 * scale_y)
            x2_img = int(x2 * scale_x)
            y2_img = int(y2 * scale_y)
            
            # 绘制矩形 - 使用更醒目的颜色和样式
            cv2.rectangle(display_frame, (x1_img, y1_img), (x2_img, y2_img), (0, 162, 232), 3)
            
            # 绘制调整点
            corner_size = 8
            cv2.rectangle(display_frame, (x1_img - corner_size, y1_img - corner_size), 
                         (x1_img + corner_size, y1_img + corner_size), (255, 0, 0), -1)  # 左上角
            cv2.rectangle(display_frame, (x2_img - corner_size, y1_img - corner_size), 
                         (x2_img + corner_size, y1_img + corner_size), (255, 0, 0), -1)  # 右上角
            cv2.rectangle(display_frame, (x1_img - corner_size, y2_img - corner_size), 
                         (x1_img + corner_size, y2_img + corner_size), (255, 0, 0), -1)  # 左下角
            cv2.rectangle(display_frame, (x2_img - corner_size, y2_img - corner_size), 
                         (x2_img + corner_size, y2_img + corner_size), (255, 0, 0), -1)  # 右下角
            
            # 绘制边界中点
            mid_top_x = (x1_img + x2_img) // 2
            mid_bottom_x = mid_top_x
            mid_left_y = (y1_img + y2_img) // 2
            mid_right_y = mid_left_y
            
            cv2.rectangle(display_frame, (mid_top_x - corner_size, y1_img - corner_size), 
                         (mid_top_x + corner_size, y1_img + corner_size), (255, 0, 0), -1)  # 上边界中点
            cv2.rectangle(display_frame, (mid_bottom_x - corner_size, y2_img - corner_size), 
                         (mid_bottom_x + corner_size, y2_img + corner_size), (255, 0, 0), -1)  # 下边界中点
            cv2.rectangle(display_frame, (x1_img - corner_size, mid_left_y - corner_size), 
                         (x1_img + corner_size, mid_left_y + corner_size), (255, 0, 0), -1)  # 左边界中点
            cv2.rectangle(display_frame, (x2_img - corner_size, mid_right_y - corner_size), 
                         (x2_img + corner_size, mid_right_y + corner_size), (255, 0, 0), -1)  # 右边界中点
        
        # 转换为Qt可显示的格式
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))
    
    def change_tracker(self, text):
        """更改跟踪器类型"""
        tracker_map = {
            'dimp (默认)': ('dimp', 'dimp50'),
            'atom': ('atom', 'default'),
            'kys': ('kys', 'default'),
            'eco': ('eco', 'default'),
            'segm': ('segm', 'default')
        }
        
        self.tracker_name, self.tracker_param = tracker_map.get(text, ('dimp', 'dimp50'))
        self.status_label.setText(f'状态: 已选择跟踪器 - {text}')
    
    def create_tracker(self):
        """创建跟踪器实例"""
        try:
            model_path = r"D:\down\SAM2APP\models\PromptVT\PromptVT.pth"
            params = parameters(yaml_name=r"D:\down\SAM2APP\components\PromptVT\experiments\PromptVT\baseline.yaml")
            self.tracker = PromptVT(params, 'video', model_path)
        except Exception as e:
            self.status_label.setText(f'状态: 创建跟踪器失败 - {str(e)}')
            # 创建一个模拟的跟踪器，以便程序可以继续运行
            self.tracker = Tracker(self.tracker_name, self.tracker_param)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.tracking_thread and self.tracking_thread.isRunning():
            self.tracking_thread.stop()
            
        if self.video_capture:
            self.video_capture.release()
            
        event.accept()

if __name__ == '__main__':
    # 启用高DPI支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    
    app = QApplication(sys.argv)
    window = ObjectTrackerApp()
    window.show()
    sys.exit(app.exec_())    