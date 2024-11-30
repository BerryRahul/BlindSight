import cv2
import numpy as np
import urllib.request
import os

url = #route

cap = cv2.VideoCapture(url)
whT=320
confThreshold = 0.5
nmsThreshold = 0.3
classesfile='coco.names'
classNames=[]

def findObject(outputs, im):
    hT, wT, cT = im.shape
    bbox = []
    classIds = []
    confs = []
    found_cat = False
    found_bird = False

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    print(indices)

    if len(indices) > 0:
        for i in indices.flatten():  
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            if classNames[classIds[i]] == 'bird':
                found_bird = True
            elif classNames[classIds[i]] == 'cat':
                found_cat = True
            
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(im, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    else:
        print("No objects detected with sufficient confidence.")


       

i = 0
while True:

    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    im = cv2.imdecode(imgnp,-1)
    if im is not None and i%9==0:
        print(i)
        print('Saving image!')
        cv2.imwrite(f'./saved_imgs/{i}.png', im)
        os.system(f'scp ./saved_imgs/{i}.png ubuntu@195.242.13.247:/home/ubuntu/meta_llama_hacks/inputs')
    i+=1
    print(i)    
    cv2.imshow('IMage',im)
    cv2.waitKey(1)
