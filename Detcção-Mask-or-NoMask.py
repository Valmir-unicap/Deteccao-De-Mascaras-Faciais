import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
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
layer_names = [
    yolo_net.getLayerNames()[i - 1] for i in yolo_net.getUnconnectedOutLayers()
]

# Configuração para GPU (se disponível)
yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Carregar modelo MobileNetV2 para detecção de máscara
mask_model = models.mobilenet_v2(pretrained=False)
mask_model.classifier[1] = nn.Linear(
    mask_model.last_channel, 2
)  # 2 classes: máscara e sem máscara
mask_model.load_state_dict(torch.load("mobilenet_mask_classifier.pth"))
mask_model.eval()

# Transformações para pré-processamento da imagem
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# Função para detectar rostos usando YOLO-Face
def detect_faces_yolo(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False
    )
    yolo_net.setInput(blob)
    layer_outputs = yolo_net.forward(layer_names)

    boxes = []
    confidences = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.6 and class_id == 0: 
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

    return result


# Função para classificar rostos como com ou sem máscara
def classify_mask(face_img):
    face_tensor = transform(face_img).unsqueeze(0)
    with torch.no_grad():
        output = mask_model(face_tensor)
        _, predicted = torch.max(output, 1)
    return "Mask" if predicted.item() == 0 else "No Mask"

current_camera = 0
video_stream = VideoStream(src=current_camera).start()

while True:
    frame = video_stream.read()
    if frame is None or not video_stream.ret:
        print("Erro ao capturar o frame.")
        break

    frame = cv2.flip(frame, 1)

    # Detectar rostos
    faces = detect_faces_yolo(frame)

    for x, y, w, h in faces:
        face_img = frame[y : y + h, x : x + w]
        if face_img.size == 0:  # Verificar se o recorte é válido
            continue

        label = classify_mask(face_img)
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Mostrar o frame
    cv2.imshow("Detecção de Rosto e Máscara", frame)

    # Encerrar com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Encerrar o stream
video_stream.stop()
cv2.destroyAllWindows()
