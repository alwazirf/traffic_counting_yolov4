import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import math
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import pyshine as ps
import cv2, imutils
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
# 
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PyQt5.QtGui import (QImage, QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PyQt5.QtWidgets import *

# SPLASH SCREEN
from splash_screen import Ui_SplashScreen

# MAIN WINDOW
from ui_main import Ui_MainWindow

# GLOBALS
counter = 0
GLOBAL_STATE = False
File = None
car = None
truck = None

# MAIN WINDOW CLASS
class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # self.maximize_restore()
        #close button
        self.ui.closeAppBtn.clicked.connect(lambda: self.close())
        #minimize button
        self.ui.minimizeAppBtn.clicked.connect(lambda: self.showMinimized())
        #maxsimize/restore
        self.ui.maximizeRestoreAppBtn.clicked.connect(lambda: self.maximize_restore())
        #pilih file
        self.ui.pushButton.clicked.connect(lambda: self.open_img())
        self.ui.pushButton_3.clicked.connect(lambda: self.start())

        self.center()

    def center(self):
        setGeometry = self.frameGeometry()
        centerGeometry = QDesktopWidget().availableGeometry().center()
        setGeometry.moveCenter(centerGeometry)
        self.move(setGeometry.topLeft())
        
    def maximize_restore(self):
        global GLOBAL_STATE
        status = GLOBAL_STATE
        if status == False:
            self.showMaximized()
            GLOBAL_STATE = True
            self.ui.maximizeRestoreAppBtn.setToolTip("Restore")
            self.ui.maximizeRestoreAppBtn.setIcon(QIcon(u"data/icons/icon_restore.png"))
        else:
            GLOBAL_STATE = False
            self.showNormal()
            self.ui.maximizeRestoreAppBtn.setToolTip("Maximize")
            self.ui.maximizeRestoreAppBtn.setIcon(QIcon(u"data/icons/icon_maximize.png"))
    
    def start(self):
        global File
        if File == None:
            print("File not found")
        else:
            self.run(File)
    
    def open_img(self):
        global File
        fname, filter = QFileDialog.getOpenFileName(self,'Open File', "", "Excel File(*.mp4 or *.avi)")
        if fname:
            # self.loadImage(fname)
            self.frame = cv2.frame = cv2.imread(fname)
            self.tmp = self.frame
            self.ui.plainTextEdit.setPlainText(fname)
            File = fname
            return File

        else:
            print("File yang diilih tidak dapat diproses atau file tidak ada")
        if len(fname) >0:
            file_path=fname.replace('/','\\')
            # self.start(fname)
            
    def display_frame(self,frame):
        global GLOBAL_STATE
        status = GLOBAL_STATE
        if status == False:
            frame = imutils.resize(frame,width=920, height=880)
        else:
            frame = imutils.resize(frame,width=1500, height=1400)
        qformat = QImage.Format_Indexed8
        if len(frame.shape) == 3:
            if (frame.shape[2] == 4):
                qformat = QImage.Format_RGBA8888
                print('qimage')
            else:
                qformat =QImage.Format_RGB888
        # create QImage from image
        frame = QImage(frame,frame.shape[1],frame.shape[0],qformat)
        frame = frame.rgbSwapped()
        # show image in img_label
        self.ui.label.setPixmap(QPixmap.fromImage(frame))
                
    def run(self,catch):
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
        def vector_angle(midpoint, previous_midpoint):
            x = midpoint[0] - previous_midpoint[0]
            y = midpoint[1] - previous_midpoint[1]
            return math.degrees(math.atan2(y, x))

        global truck
        global car
        titik1=(100,511)
        titik2=(551,511)
        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        nms_max_overlap = 1.0

        # initialize deep sort
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        tracker = Tracker(metric)

        # initialize counting variables
        count_dict = {}  # initiate dict for storing counts
        total_counter = 0
        up_count = 0
        down_count = 0
        from collections import Counter
        class_counter = Counter()  # store counts of each detected class
        from collections import deque
        already_counted = deque(maxlen=50)  # temporary memory for storing counted IDs
        intersect_info = []  # initialise intersection list
        memory = {}

        # load configuration for object detector
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        input_size = 416
        video_path = 'C:/Users/MSI Laptop/Pictures/overpass.mp4' #ini dia

        saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

        # begin video capture
        try:
            vid = cv2.VideoCapture(int(catch))
        except:
            vid = cv2.VideoCapture(catch)

        frame_num = 0
        # while video is running
        while True:
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                print('Video Telah Selesai atau Gagal Memuat, coba dengan Video lainnya!')
                break
            frame_num +=1
            # print('Frame #: ', frame_num)
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.45,
                score_threshold=0.50
            )

            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())
            
            # custom allowed classes (uncomment line below to customize tracker for only people)
            # allowed_classes = ['person']

            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)

            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)
            #buat garis biru
            cv2.line(frame,titik1,titik2, (0, 255, 255), 2)

            # update tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr() # Get current position in bounding box format `(min x, miny, max x,max y)
                #track_cls = track.cls  # most common detection class for track
                class_name = track.get_class()
                
                #Object counting
                midpoint = track.tlbr_midpoint(bbox) # Finds midpoint of a box in tlbr format.
                origin_midpoint = (midpoint[0], frame.shape[0] - midpoint[1])  # get midpoint respective to botton-left

                if track.track_id not in memory:
                    memory[track.track_id] = deque(maxlen=2)
                
                memory[track.track_id].append(midpoint)
                previous_midpoint = memory[track.track_id][0]

                origin_previous_midpoint = (previous_midpoint[0], frame.shape[0] - previous_midpoint[1])

                if intersect(midpoint, previous_midpoint, titik1,titik2) and track.track_id not in already_counted:
                    class_counter[class_name] += 1
                    total_counter += 1
                    cv2.line(frame,titik1,titik2, (255, 0, 0), 2) #garis merah
                    already_counted.append(track.track_id)  # Set already counted for ID to true.
                    angle = vector_angle(origin_midpoint, origin_previous_midpoint)

                    if angle > 0:
                        up_count += 1
                    if angle < 0:
                        down_count += 1
                # draw bbox on screen
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            if len(memory) > 50:
                del memory[list(memory)[0]]

            fps = 1.0 / (time.time() - start_time)
            # Draw total count.
            text = ("FPS: %.2f" %fps)
            frame =  ps.putBText(frame,text,text_offset_x=int(frame.shape[1]-185),text_offset_y=int(0.05 * frame.shape[0]),vspace=10,hspace=10, font_scale=1.0,background_RGB=(228,20,222),text_RGB=(255,255,255))
            text = "Total: {}".format(str(total_counter))
            frame =  ps.putBText(frame,text,text_offset_x=int(10),text_offset_y=int(0.05 * frame.shape[0]),vspace=10,hspace=10, font_scale=1.0,background_RGB=(10,20,222),text_RGB=(255,255,255))

            # display counts for each class as they appear
            y = 0.12 * frame.shape[0]
            for cls in class_counter:
                class_count = class_counter[cls]
                text = str(cls) + " " + str(class_count)
                if str(cls) == 'car':
                    car = str(class_count)
                elif str(cls) == 'truck':
                    truck = str(class_count)
                frame =  ps.putBText(frame,text,text_offset_x=int(10),text_offset_y=int(y),vspace=5,hspace=10, font_scale=1.0,background_RGB=(20,210,4),text_RGB=(255,255,255))
                y += 0.05 * frame.shape[0]
            # self.ui.label_2.setText(text)
            # calculate frames per second of running detections
            # fps = 1.0 / (time.time() - start_time)
            # print("FPS: %.2f" % fps)
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # cv2.imshow("Output Video", result)
            # self.ui.label.setPixmap(QPixmap.fromImage(result))
            self.display_frame(result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()
            
# SPLASH SCREEN CLASS
class SplashScreen(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_SplashScreen()
        self.ui.setupUi(self)

        # REMOVE TITLE BAR
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # DROP SHADOW EFFECT
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 60))
        self.ui.dropShadowFrame.setGraphicsEffect(self.shadow)

        # QTIMER ==> START
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.progress)
        # TIMER IN MILLISECONDS
        self.timer.start(45)

        # Initial Text
        self.ui.label_description.setText("YOLO Traffic")
        self.ui.label_loading.setText("Please Wait")

        # Change Texts
        QtCore.QTimer.singleShot(1500, lambda: self.ui.label_loading.setText("Loading Assets"))
        QtCore.QTimer.singleShot(3000, lambda: self.ui.label_loading.setText("Loading User Interface"))

        self.center()
        self.show()

    # APP FUNCTIONS
    def progress(self):

        global counter

        # SET VALUE TO PROGRESS BAR
        self.ui.progressBar.setValue(counter)

        # CLOSE SPLASH SCREEN AND OPEN APP
        if counter > 100:
            # STOP TIMER
            self.timer.stop()

            # SHOW MAIN WINDOW
            self.main = MainWindow()
            self.main.show()

            # CLOSE SPLASH SCREEN
            self.close()

        # INCREASE COUNTER
        counter += 1

    def center(self):
        setGeometry = self.frameGeometry()
        centerGeometry = QDesktopWidget().availableGeometry().center()
        setGeometry.moveCenter(centerGeometry)
        self.move(setGeometry.topLeft())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SplashScreen()
    sys.exit(app.exec_())
