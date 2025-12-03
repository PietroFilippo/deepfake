import cv2 
from    threading import Thread 
import time 

class WebcamStream: 
    def __init__(self, src=0, fps=30): 
        # Usa DirectShow (CAP_DSHOW) para Windows para evitar erros MSMF e melhorar a velocidade de inicialização 
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW) 
        (self.grabbed, self.frame) = self.stream.read() 
        self.stopped = False 
        # Define resolução, pode ser ajustado 
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) 
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) 
        self.stream.set(cv2.CAP_PROP_FPS, fps)

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

class VideoFileStream:
    def __init__(self, path):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.frame_count = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)

    def start(self):
        return self

    def read(self):
        if self.stopped:
            return None
        (grabbed, frame) = self.stream.read()
        if not grabbed:
            return None
        return frame

    def stop(self):
        self.stopped = True
        self.stream.release()