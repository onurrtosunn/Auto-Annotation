import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm
import glob

os.chdir("/path/to/target/dir")
IMAGE_PATH = "images/"
OUTPUT_IMAGES = "output_images/"
OUTPUT_LABELS = "output_labels/"
label_list = open("obj.names").read().strip().split("\n")
weight_path = "yolov4.weights"
config_path = "yolov4.cfg"


def convert_pascal_to_yolo(size, box):
    """
    Convert Pascal VOC to Yolo format. When converting bounding boxes, 
    you can modify pixels manually. Bounding box success depends on model accuracy
    """
    dw = 1./size[0]
    dh = 1./size[1]
    x = ((box[0]-2) + box[1])/2.0  
    y = ((box[2]-2)+ box[3])/2.0
    w = box[1] - box[0] + 2
    h = box[3] - box[2] + 2

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x,y,w,h)

def draw_bounding_box(obj_list, img_path, dest_path, labels_path):
    """
    Draw bounding boxes on image. Results saving 
    another folder
    """
    colors = {
        0: (0, 0, 255),
        1: (255, 255, 0),
        2: (0, 255, 0),
        3: (255, 21, 10),
        4: (120, 97, 45),
        5: (127, 0, 255),
        6: (244, 244, 0),
        7: (32, 255, 244),
    }

    for images in tqdm(list(glob.iglob(os.path.join(img_path, "*.jpg")))):
        img_name, ext = os.path.splitext(os.path.basename(images))
        img = cv2.imread(f'{img_path}/{img_name}.jpg')
        dh, dw, _ = img.shape
        label_file = open(f'{labels_path}/{img_name}.txt', 'r')
        label_data = label_file.readlines()
        label_file.close()

        for dt in label_data:
            line = dt.split(' ')
            if line[-1] == "\n":  # in case, EOF is '\n'
                line = line[:-1]
            c, x, y, w, h = map(float, line)
            left = int((x - w / 2) * dw)
            right = int((x + w / 2) * dw)
            top = int((y - h / 2) * dh)
            bottom = int((y + h / 2) * dh)

            if left < 0:
                left = 0
            if right > dw - 1:
                right = dw - 1
            if top < 0:
                top = 0
            if bottom > dh - 1:
                bottom = dh - 1

            color = colors[int(c)]
            cv2.rectangle(img, (left, top), (right, bottom), color, 1)
            cv2.putText(img, obj_list[int(c)], (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imwrite(f'{dest_path}/{img_name}.jpg', img)

def detect_and_create_labels(label_list,image_folder,dest_path,labels_path):

    net = cv2.dnn.readNetFromDarknet(config_path, weight_path)
    path_df = []
    class_total_df = []
    class_df = []
    xmin_df = []
    xmax_df = []
    ymin_df = []
    ymax_df = []
    confidence_df = []
    x_df = []
    y_df = []
    w_df = []
    h_df = []
    H_df = []
    W_df = []

    for images in os.listdir(image_folder):
        image = cv2.imread(image_folder + images)
        (H, W) = image.shape[:2]
        
        ln = net.getLayerNames()
        ln=[ln[i-1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                    swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
        file = open(labels_path + os.path.splitext(images)[0]+".txt", "w")
        size = W*H

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                xmin = x
                xmax = x + w
                ymin = y
                ymax = y + h

                b = (float(xmin), float(xmax), float(ymin), float(ymax))
                x_yolo,y_yolo,w_yolo,h_yolo = convert_pascal_to_yolo((W,H), b)

                path_df += [image_folder]
                class_total_df += [len(idxs)]
                class_df += [classID]
                xmin_df += [xmin]
                xmax_df += [xmax]
                ymin_df += [ymin]
                ymax_df += [ymax]
                x_df += [x_yolo]
                y_df += [y_yolo]
                w_df += [w_yolo]
                h_df += [h_yolo]
                W_df += [W]
                H_df += [H]
                confidence_df += [confidences[i]]
                data_txt =[]

                data_txt.append(str(classID)+" "+str(x_yolo)+" "+str(y_yolo)+" "+str(w_yolo)+" "+str(h_yolo))

                data = pd.DataFrame({'path': path_df, 'width': W_df, 'height': H_df ,'total_class': class_total_df, \
                                            'class_object' : class_df, 'xmin': xmin_df, 'xmax': xmax_df, 'ymin': ymin_df, 'ymax': ymax_df, \
                                            'x': x_df, 'y':y_df, 'w': w_df, 'h': h_df, 'confidence': confidence_df})
                listToStr = ' '.join(map(str, data_txt))
                file.write(listToStr+'\n')


    data.to_csv('data.csv',index=False)
    file.close()

def create_folder():
    try:
        if not os.path.exists("output_labels"):
            print("Ouput labels folder creating...")
            os.makedirs("output_labels")
        if not os.path.exists("output_images"):
            print("Ouput images folder creating...")
            os.makedirs("output_images")
    except OSError:
        print ('Error: Creating directory. ')

create_folder()
detect_and_create_labels(label_list, IMAGE_PATH, OUTPUT_IMAGES, OUTPUT_LABELS)
draw_bounding_box(label_list, IMAGE_PATH, OUTPUT_IMAGES, OUTPUT_LABELS)