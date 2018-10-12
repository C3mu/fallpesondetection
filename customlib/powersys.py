import socket
import time
from threading import Thread

class powersys:
    def __init__(self,positionon,positionoff,num,number,HOST,PORT):
        '''
        #expor TABLE
        #1:nowmal on/off with position
        #2:lock auto control
        '''
        print("my number:",number)
        self.HOST = HOST
        self.PORT = PORT
        self.expor = 1
        self.timeon = time.time()
        self.numupdate = 0
        self.stage = False
        self.num = num
        self.number = number
        self.positionon = positionon
        self.positionoff = positionoff
        self.laststage = 0

    def on(self):
        self.expor = 2
        self.stage = True

    def off(self):
        self.expor = 2
        self.stage = False
 
    def retn(self):
        return self.stage

    def renum(self):
        return self.num
    
    def senddata(self, data):
        if self.laststage <= 5:
            t0 = Thread(target=self.startsend, args=(data,))
            t0.daemon = True
            t0.start()
            self.laststage += 1
    
    def startsend(self, data):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.HOST, self.PORT))
        print("swich ",data)
        s.send(data)
        s.close()

    def check(self, inputi,number): # base function
        if self.number == number:
            #print("activate:",number)
            if self.expor == 1:   
                x, y, w, h = inputi
                a,b,c,d = self.positionon
                if a < x and w < c and b <= y and h <= d:
                        self.senddata(bytearray(str(self.num)+',255','utf-8')) #turn on when in 'on' position
                        self.stage = True
                        if self.laststage <= 6:
                            self.laststage += 1
                        self.timeon = time.time()
                if time.time() - self.timeon > 2:
                    self.timeon = time.time()
                    (x, y, w, h) = self.positionoff
                    if not (a < x and w < c and b < y and h < d):
                        if self.laststage >= 0:
                            self.senddata(bytearray(str(self.num)+',0','utf-8')) #turn off when out 'off' position
                            self.laststage -= 2
                            self.stage = False
