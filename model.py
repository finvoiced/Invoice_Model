import cv2
import numpy as np
import json
import Train
from pymongo import MongoClient
import pandas as pd

client = MongoClient('mongodb+srv://Finvoiced-cluster001:Finvoiced1234@finvoiced-cluster001.xjlgm.mongodb.net/myFirstDatabase?retryWrites=true&w=majority',tls=True, tlsAllowInvalidCertificates=True )
db = client.get_database('finvoiced_db')
records = db.information_records

file_name = '/content/gdrive/MyDrive/yolov4/image/image101.jpg'
config = '/content/gdrive/MyDrive/yolov4/yolov4-obj_51.cfg'
weights = '/content/gdrive/MyDrive/yolov4/Weights_51/yolov4-obj_best_51.weights'
classes_ = '/content/classes.txt'

def get_output_layers(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def convert_coordinates(size, box):
    dw = 1.0/size[0]
    dh = 1.0/size[1]
    x = (box[0]+box[1])/2.0
    y = (box[2]+box[3])/2.0
    w = box[1]-box[0]
    h = box[3]-box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
#image = cv2.imread(args.image)
image = cv2.imread(file_name)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open(classes_, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

#net = cv2.dnn.readNet(args.weights, args.config)
net = cv2.dnn.readNet(weights, config)

blob = cv2.dnn.blobFromImage(image, scale, (608,608), (0,0,0), True, crop=False)

net.setInput(blob)

outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4


for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])


indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)


with open("yolo.txt", "w") as f:
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        b = (float(x), float(x+w), float(y), float(y+h))
        bb = convert_coordinates((Width,Height), b)
        f.write(str(class_ids[i]) + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()


invoice_dict = Train.Train(file_name,'yolo.txt')
with open('data.json', 'w') as fp:
    json.dump(invoice_dict, fp)
try:
  x = records.insert_one(invoice_dict)
  print(" Result Stored in MongoDB")
except:
  print("An exception occurred in MongoDB")

