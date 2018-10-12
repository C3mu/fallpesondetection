import os
import sys
import tensorflow as tf
import time
import cv2
import requests
from threading import Thread
import numpy as np

sys.path.append("..")
sys.path.append("..")
from utils import label_map_util # to get the data
from utils import visualization_utils as vis_util #for labling the on image

class MAINStream:
    def __init__(self,MODEL_NAME,HOST,VISUAL=True,CUSTOM=False,SET=None):
        #load pack
        self.MODEL_NAME = MODEL_NAME
        self.PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

        #load host to get img
        self.HOST = HOST.copy()
        self.r = []

        #control
        self.VISUAL = VISUAL
        self.stopped = False
        self.heygofaster = None
        self.pesonhere = []
        self.timeon = []

        #image
        self.boxes = None
        self.frame = []
        self.lastframe = None

        #check all connection
        for n,i in enumerate(self.HOST):
            if CUSTOM:
                requests.get('http://'+i+':8080/'+SET[n],auth=('user','password'), stream=True)
            self.frame.append(None)
            self.r.append(None)
            self.pesonhere.append(False)
            self.timeon.append(time.time())
            print('connecting to http://'+i+':8080/stream/video.mjpeg')
            try:
                self.r[n] = requests.get('http://'+i+':8080/stream/video.mjpeg',auth=('user','password'), stream=True)
            except:
                print("Failes to connect ",i,n)
                sys.exit(1)
            if self.r[n].status_code != 200:
                print("Failes to connect ",i,n)
                sys.exit(1)
            print("oke")
        
        #inalazing ...
        self.heygofaster = self.pesonhere.copy()
        self.lastframe = self.frame.copy()
        self.boxes = self.frame.copy()

    def detectingmoving(self,num):
                gray = cv2.cvtColor(self.frame[num],cv2.COLOR_BGR2GRAY).copy()
                gray = cv2.GaussianBlur(gray,(11,11),0)
                cv2.accumulateWeighted(gray,self.lastframe[num],0.3)
                frameDelta = cv2.absdiff(gray,cv2.convertScaleAbs(self.lastframe[num]))
                thresh = cv2.threshold(frameDelta,10,255,cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=6)
                cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[1]
                for c in cnts:
                    if cv2.contourArea(c) > 333:
                            (x, y, w, h) = cv2.boundingRect(c)
                            if not self.pesonhere[num]:
                                self.heygofaster[num] = True
                                self.timeon[num] = time.time()
                            break
                if time.time() - self.timeon[num] > 1:
                    self.heygofaster[num] = False # mean ohh i dont wana go faster ;D

    def updatecamera(self,num):
        bytees = bytes()
        for chunk in self.r[num].iter_content(chunk_size=1024):
            bytees += chunk
            a = bytees.find(b'\xff\xd8')
            b = bytees.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytees[a:b+2]
                bytees = bytees[b+2:]
                self.frame[num] = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.gotrack(num)
            if self.stopped:
                break

    def gotrack(self,num):
            if self.lastframe[num] is None:
                gray = cv2.cvtColor(self.frame[num],cv2.COLOR_BGR2GRAY).copy()
                self.lastframe[num]  = gray.copy().astype("float")
            t = Thread(target=self.detectingmoving, args=(num,))
            t.daemon = True
            t.start()

    def updatedetection(self):#SETBYSELF CODE !!! :D cpy paste is fastest than use for loop ??? ok
        detection_graph = tf.Graph()
        PATH_TO_LABELS = self.MODEL_NAME+"/object-detection.pbtxt"
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)
            boxess = detection_graph.get_tensor_by_name('detection_boxes:0')
            scoress = detection_graph.get_tensor_by_name('detection_scores:0')
            classess = detection_graph.get_tensor_by_name('detection_classes:0')
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            num_detectionss = detection_graph.get_tensor_by_name('num_detections:0')

            while True:
                for num,i in enumerate(self.HOST):
                    image_np = self.frame[num]
                    im_height, im_width, chanol = image_np.shape
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    (boxes, scores, classes, num_detections) = sess.run(
                    [boxess, scoress, classess, num_detectionss],
                    feed_dict={image_tensor: image_np_expanded})

                    if self.VISUAL: # some time Fasle is True
                        vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=20,
                        line_thickness=4,
                        min_score_thresh=.4)

                    # Visualization of the results of a detection.
                    boxes = np.squeeze(boxes)
                    classes = np.squeeze(classes).astype(np.int32), # "," <- donot delete && still dont know why
                    scores = np.squeeze(scores),
                    trackperson = []
                    
                    for i in range(0,boxes.shape[0]):#check all data back
                        if (scores[0][i] > 0.5): #if high pp 
                            if classes[0][i] == 1: #if that is person '1'
                                ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
                                a = (xmin * im_width, ymin * im_height, xmax * im_width, ymax * im_height)
                                trackperson.append(a)
                    if trackperson != []:
                        self.pesonhere[num] = True
                    else:
                        self.pesonhere[num] = False
                        trackperson = None
                    self.boxes[num] = trackperson

                    if not self.heygofaster[num]:
                        #print("start blink")
                        x = 0
                        while x <= 40: #update every two second !!!
                            x+=1
                            time.sleep(0.05)
                            if self.heygofaster[num] == True:
                                break
                    if self.stopped == True:
                        break

    def readAI(self):
        return self.boxes.copy()

    def readcam(self):
        return self.frame.copy()

    def stop(self):
        self.stopped = True
                    
    def start(self):
        for num,x in enumerate(self.HOST):
            t0 = Thread(target=self.updatecamera, args=[num])
            t0.daemon = True
            t0.start()
        t1 = Thread(target=self.updatedetection, args=())
        t1.daemon = True
        time.sleep(2)
        t1.start()
        print("wait for detection pack ...")
        time.sleep(10)
        print("sys in work now :) ")
