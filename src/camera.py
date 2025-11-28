import cv2 
from    threading import Thread 
import time 

class WebcamStream: 
    def __init__(self, src=0): 
        # Usa DirectShow (CAP_DSHOW) para Windows para evitar erros MSMF e melhorar a velocidade de inicialização 
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW) 
        (self.grabbed, self.frame) = self.stream.read() 
        self.stopped = False 
        # Define resolução mais alta se possível, pode ser ajustado 
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
    def start(self): 
        Thread(target=self.update, args=()).start() 
        return self 
    def update(self): 
        while True: 
            if self.stopped: 
                return 
            (self.grabbed, self.frame) = self.stream.read() 
    def read(self): 
        return self.frame 
    def stop(self): 
        self.stopped = True 
        self.stream.release()