import cv2
import numpy as np
import urllib.request
import time
import os

# Camera and Model Configuration
url = 'http://192.168.155.123/cam-hi.jpg'
whT = 320
confThreshold = 0.75
nmsThreshold = 0.15
sharpness_threshold = 100  # Threshold for sharpness filtering
classesfile = 'coco.names'
classNames = []
with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    
real_heights = {
    'person': 1, 
    'cat': 0.3,     
    'dog': 0.5,    
    'bird': 0.1,    
    'bottle': 0.2,
    'laptop': 0.3,
    'tvmonitor': 0.3,
    'chair': 0.5,
    'table': 0.7,
    'diningtable': 0.7
}

modelConfig = 'yolov3.cfg'
modelWeights = 'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


save_interval = 0.1 
duration = 5  
frame_count = 0
output_folder = './saved_frames'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to detect objects and display distances
def findObject(outputs, im):
    hT, wT, cT = im.shape
    bbox = []
    classIds = []
    confs = []

    f_mm = 3.6  # Focal length in mm
    sensor_width_mm = 4.8  # Sensor width in mm
    f_pixels = (f_mm * wT) / sensor_width_mm  # Focal length in pixels

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
    if len(indices) > 0:
        for i in indices.flatten():
            box = bbox[i]
            x, y, w, h = box
            object_name = classNames[classIds[i]]

            if h > 0:
                if object_name in real_heights:  
                    H_real = real_heights[object_name]
                    distance = (H_real * f_pixels) / h
                    distance_text = f"{distance:.2f}m"
                    print(f"{object_name}: Distance = {distance_text}")
                else:
                    distance_text = "N/A"
                    print(f"Height not defined for {object_name}, skipping distance calculation")
            else:
                distance_text = "Invalid"
                print(f"Invalid bounding box height for {object_name}")

            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(
                im,
                f'{object_name.upper()} {int(confs[i] * 100)}% Dist: {distance_text}',
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )


def is_sharp(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var > sharpness_threshold, laplacian_var

start_time = time.time()
last_saved_time = start_time

sharpest_frame = None
max_sharpness = 0

while True:
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    im = cv2.imdecode(imgnp, -1)

    if im is None:
        print("Failed to retrieve frame")
        break

    blob = cv2.dnn.blobFromImage(im, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layernames = net.getLayerNames()
    outputNames = [layernames[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    outputs = net.forward(outputNames)
    findObject(outputs, im)

    sharp, sharpness_value = is_sharp(im)
    print(f"Sharpness: {sharpness_value:.2f}")

    if sharpness_value > max_sharpness:
        max_sharpness = sharpness_value
        sharpest_frame = im.copy()
        print(f"New sharpest frame with sharpness {max_sharpness:.2f}")
    
    current_time = time.time()
    if (current_time - last_saved_time >= save_interval):
        frame_count += 1
        frame_path = os.path.join(output_folder, f'frame_{frame_count}.jpg')
        # cv2.imwrite(frame_path, im)
        print(f"Saved sharp frame: {frame_path}")
        last_saved_time = current_time
    
    if time.time() - start_time >= duration:
        print("Frame capturing complete.")
        break

    cv2.imshow('Image', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if sharpest_frame is not None:
    sharpest_frame_path = os.path.join(output_folder, f'sharpest_frame.png')
    cv2.imwrite(sharpest_frame_path, sharpest_frame)
    os.system(f'scp ./{output_folder}/sharpest_frame.png ubuntu@195.242.13.247:/home/ubuntu/meta_llama_hacks/inputs')   
    print(f"Sharpest frame saved at: {sharpest_frame_path}")
else:
    print("No frames were sharp enough to save.")

cv2.destroyAllWindows()
