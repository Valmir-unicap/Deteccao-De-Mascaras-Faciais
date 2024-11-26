import cv2
import numpy as np
from threading import Thread

# Classe para captura de vídeo em threads
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.stream.isOpened():
                self.stop()
            self.ret, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


# Carregar YOLO-Face
yolo_net = cv2.dnn.readNet("face-yolov3.weights", "face-yolov3-tiny.cfg")
layer_names = [yolo_net.getLayerNames()[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Configuração para GPU (se disponível)
yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Função para detectar rostos usando YOLO-Face
def detect_faces_yolo(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    layer_outputs = yolo_net.forward(layer_names)
    
    boxes = []
    confidences = []
    
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5 and class_id == 0:  # Filtrar apenas faces
                box = detection[0:4] * np.array([width, height, width, height])
                center_x, center_y, box_width, box_height = box.astype("int")
                x = int(center_x - (box_width / 2))
                y = int(center_y - (box_height / 2))
                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    result = []
    if indices is not None and len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            result.append((x, y, w, h))
    else:
        print("Nenhuma face detectada.")
    
    return result


# Iniciar o stream de vídeo
video_stream = VideoStream(src=0).start()

while True:
    frame = video_stream.read()
    if frame is None or not video_stream.ret:
        print("Erro ao capturar o frame.")
        break

    # Detectar rostos
    faces = detect_faces_yolo(frame)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Rosto", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar o frame
    cv2.imshow("Detecção de Rosto com YOLO-Face", frame)

    # Encerrar com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Encerrar o stream
video_stream.stop()
cv2.destroyAllWindows()
