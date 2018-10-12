import numpy as np
import os
import sys
import tensorflow as tf
import pickle
import time
from customlib.sendtopi import sendtopi

##from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import PIL.ImageDraw as ImageDraw
#This is needed since the notebook is stored in the object_detection folder. mean back to
sys.path.append("..")
# ## Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util # to get the data
from utils import visualization_utils as vis_util #for labling the ...idk... on image
#just for show and work on image
import cv2
# to not off line
import socket
import requests
from imutils.video import FPS # test the FPS
# ...multytheading :D
from threading import Thread

##Model preparation
#MODEL = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL = 'fallpersondetec900000'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL + '/frozen_inference_graph.pb'
# List of the strings that is used
#to add correct label for each box.
PATH_TO_LABELS = MODEL+"/object-detection.pbtxt"#os.path.join('data', 'mscoco_label_map.pbtxt')
#PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 3

#variable
host = '192.168.0.101' # ip of raspberry pi

#camip = '192.168.0.101' # ip of raspberry pi
SETCUSTOM = 'http://'+host+':8080/panel?width=720&height=720&format=1196444237&9963776=36&9963777=-28&9963778=-100&9963790=100&9963791=100&9963803=-74&9963810=0&134217728=0&134217729=1&134217730=0&134217739=10&134217741=10&9963796=0&9963797=0&134217734=0&134217736=0&134217737=1&134217738=0&134217740=0&134217731=0&134217732=1&134217733=0&134217735=3&apply_changed=1'

#rightcolor#http://192.168.137.205:8080/panel?width=1080&height=1080&format=1196444237&9963776=48&9963777=11&9963778=25&9963790=95&9963791=108&9963803=35&9963810=0&134217728=0&134217729=1&134217730=0&134217739=50&134217741=10&9963796=0&9963797=0&134217734=0&134217736=0&134217737=1&134217738=0&134217740=0&134217731=6&134217732=1&134217733=0&134217735=3&apply_changed=17
port = 12345
POSITION_DATA = {}#yes a library to save position add with what sys will do with :) ...english
SAVEFILENAME = '/save/positon'
'''#########################CLASS#########################'''
class getVideoStream:
    def __init__(self):
        self.fps = FPS().start()
        self.frame = None
        self.stopped = False
        #start request to get stream camera data
        #requests.get(SETCUSTOM, auth=('user', 'password'), stream=True)
        self.r = requests.get('http://' +host +':8080/stream/video.mjpeg', auth=('user', 'password'), stream=True)
        #self.r = requests.get('rtsp://'+camip+':554/onvif1', auth=('user', 'password'), stream=True)
        #self.r = requests.get('rtsp://192.168.0.101:554/onvif1', stream=True)#auth=('admin', 'admin')
        if self.r.status_code != 200:
              print("Failes to connect")
              exit
        print("connected to ",host)
        

    def start(self):
        t0 = Thread(target=self.update, args=())
        t0.daemon = True
        t0.start()
        #time.sleep(2)
        
    def update(self):
        bytees = bytes()
        for chunk in self.r.iter_content(chunk_size=1024):
            bytees += chunk
            a = bytees.find(b'\xff\xd8')
            b = bytees.find(b'\xff\xd9')
            if a != -1 and b != -1:
                  self.fps.update()
                  jpg = bytees[a:b+2]
                  bytees = bytees[b+2:]
                  self.frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if self.stopped:
                break
    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.fps.stop()
        print("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))

'''#########################FUNCTION########################'''
#function to connect with item v:
def readdata(): #read the data fromfile
    with open(SAVEFILENAME, 'rb') as f:
        try:
            POSITION_DATA = pickle.load(f)
            print("load file done")
        except:
            print("loadfile ERROR")
        
def savedata(): #savedata to file
    with open(POSITION_DATA, 'wb') as f:
        try:
            pickle.dump(POSITION_DATA, f)
            print("save succes")
        except:
            print("save ERROR")

def reloadclient():
    pass#program are unvailble v:

def senddata(data):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host,port))
    s.send(data)
    s.close()

def caculator(inp,adm):#inp=(x,y,w,h);adm=(xl,yl,wl,hl)
    '''#check 2 thing:
                2 insidesideposition  1 halfsideposition
                #############           #############
                ##ooooooooo##           ##ooooooooo##
                ##o#######o##           ##o#######o##
                ##o###xxx#o##           ##o#####xxxxx
                ##o###x#x#o##           ##o#####x#o#x
                ##o###xxx#o##           ##o#####x#o#x
                ##o#######o##           ##o#####x#x#x
                ##ooooooooo##           ##ooooooooo##
                #############           #############
    '''
    x,y,w,h = inp
    try:
        a,b,c,d = adm
    except:
        return 0
    re = 0
    if x > a and y > b and w < c and h < d:
        re += 2
    '''
    if x <= a and a <= w:
        re += 1
    if y <= b and b <= h:
        re += 2
    '''
    return re



def intersects(self, other):
    pass
    #if x > x1 and y > y1 and w < w1 and h < h1: # Problem
    #return not (self.top_right.x < other.bottom_left.x or self.bottom_left.x > other.top_right.x or self.top_right.y < other.bottom_left.y or self.bottom_left.y > other.top_right.y)

#start stream :D
stream = getVideoStream()
stream.start()
'''#########################MAIN'''#MAIN########################
#seting up the uv4l  
#r = requests.get('http://192.168.1.116:8080/panel?width=640&height=480&format=1196444237&9963776=50&9963777=0&9963778=0&9963790=100&9963791=100&9963803=0&9963810=0&134217728=1000&134217729=1&134217730=0&134217739=100&134217741=3&9963796=0&9963797=0&134217734=0&134217736=0&134217737=1&134217738=1&134217740=1&134217731=0&134217732=1&134217733=0&134217735=3&apply_changed=1', auth=('user', 'password'), stream=True)
# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#test
setbox = ()
sendtopi = sendtopi(host,port)
#sendtopi.senddata(b'3,0')
stage = True
isreally = 1
forsure = False
timeon = time.time()
# # Detection
fps = FPS().start()
with detection_graph.as_default():
    #Actual detection.
    sess = tf.Session(graph=detection_graph)
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    while True: 
              image_np = stream.read()
              image_np_expanded = np.expand_dims(image_np, axis=0)
              fps.update()
              # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
              # Each box represents a part of the image where a particular object was detected.
              # Each score represent how level of confidence for each of the objects.
              # Score is shown on the result image, together with the class label.
              (boxess, scoress, classess, num_detectionss) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})

              boxess = np.squeeze(boxess)
              classess = np.squeeze(classess).astype(np.int32), 
              scoress = np.squeeze(scoress),
              
              vis_util.visualize_boxes_and_labels_on_image_array(
                  image_np,
                  np.squeeze(boxess),
                  np.squeeze(classess).astype(np.int32),
                  np.squeeze(scoress),
                  category_index,
                  use_normalized_coordinates=True,
                  max_boxes_to_draw=20,
                  line_thickness=4,
                  min_score_thresh=.5)
              
              # draw to the image
              #print(isreally)
              forsure = False
              for i in range(0,boxess.shape[0]): #check all data back            
                     if (scoress[0][i] > 0.3): #if high pp
                         if classess[0][i] == 3: #if that is person '1'
                             #print(i," ",classes[0][i]," ",category_index.get(classes[0][i])," ",scores[0][i])#test
                             ymin, xmin, ymax, xmax = tuple(boxess[i].tolist())#get 'position' of that person
                             im_height, im_width, chanol = image_np.shape
                             (x, y, w, h) = (xmin * im_width, ymin * im_height, xmax * im_width, ymax * im_height)
                             isreally += 1
                             forsure = True
                             if isreally > 20:
                                 if time.time() - timeon > 5:
                                     print("detection lying person !")
                                     sendtopi.senddata(b'0,255')
                                 isreally = 20
                             cv2.rectangle(image_np,(int(x),int(y)),(int(w),int(h)),(0,255,0),3)
                         #else:
                             #if isreally > 1 :
                                 #isreally -= 1
              
              if isreally == 1:
                  sendtopi.senddata(b'255,0')
                  
              if forsure == False:
                  if isreally > 0 :
                      isreally -= 1
                  timeon = time.time()
                  
              #show the image and
              #if one == True:
              cv2.imshow('object detection',image_np) #cv2.resize(image_np, (800,600)))
              #one = False
              key = cv2.waitKey(1) & 0xFF
              if key == ord('q'):
                    stream.stop()
                    cv2.destroyAllWindows()
                    break
              if key == ord('h'):
                    sendtopi.senddata(b'255,255')
                    
              if key == ord('n'):
                    setbox = cv2.selectROI("Chose a position",image_np)#,False,False
                    cv2.destroyAllWindows()
              time.sleep(0.5)
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
