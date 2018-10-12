##this module is only use for turn off pi !!!
import socket
import time
from threading import Thread

class sendtopi:
    def __init__(self,HOST,PORT):
        self.timedeta = time.time()
        self.laststage = None
        self.HOST = HOST
        self.PORT = PORT
        pass

    def senddata(self, data):
        #if time.time() - self.timedeta > 1:# and data != self.laststage:
        #    if data != self.laststage:
                t0 = Thread(target=self.start, args=(data,))
                t0.daemon = True
                t0.start()
                self.laststage = data
                self.timedeta = time.time()

    def start(self, data):
        #try:
        #print(data)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.HOST, self.PORT))
        s.send(data)
        #print(s.recv(1024))
        s.close()
        #except:
        #print("false to connect")
        return s.recv(1024)
