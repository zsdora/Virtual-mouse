import pyautogui
import cv2
import numpy as np

# képernyő
screenWidth, screenHeight = pyautogui.size()

# webkam inic.
capture = cv2.VideoCapture(0)

# yolo model betöltése
net = cv2.dnn.readNet("./yolo_training_final.weights", "./yolov3-tiny-custom.cfg")

# egyéni objektumok osztályai
classes = ["1", '2', '3']

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

sensitivity_factor = 1.5  # érzékenység növelésére

while True:
    # kép olvasása a webkamerából
    _, frame = capture.read()
    img = cv2.resize(frame, None, fx=1, fy=1)
    height, width, channels = img.shape

    # képernyőhöz illesztés
    standWidth = int(screenWidth / width)
    standHeight = int(screenHeight / height) 

    # yolo blob létrehozása a képből
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (608, 608), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []

    # obj. detekt.
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:  # 40%-nál magasabb biztonságú detektált objektumokat kezelünk
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # egér poz. kiszám. és mozg.
    if len(boxes) > 0:
        # obj. középpontja
        center_x = (boxes[0][0] + boxes[0][2] / 2) * sensitivity_factor  # Érzékenység növ.
        center_y = (boxes[0][1] + boxes[0][3] / 2) * sensitivity_factor

        # fordított irányú korrekció (X és Y koord. csere)
        corrected_x = screenWidth - int(center_x) * standWidth  # x koord. tükrözzük
        pyautogui.moveTo(corrected_x, int(center_y) * standHeight)

    # nem-maximális elnyomás alk. (csak a legbiztosabb detektált objektumokat tartjuk meg)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.7)

    # kép frissítése, detektált objektumok körberajzolása
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = (255, 0, 0)  # kék szín bounding box-nak
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    # bounding box-ok számának alapján kattintás végrehajtása
    num_fingers = len(boxes)  # bounding box-ok száma = ujjak számának

    if num_fingers == 3:
        pyautogui.click()  # bal kattintás
    elif num_fingers == 5:
        pyautogui.rightClick()  # jobb kattintás

    # eredmény megjel.
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
