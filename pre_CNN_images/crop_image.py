import sys
sys.path.append('/root/darknet')
import darknet as dn
import cv2 as cv
import os
from matplotlib import pyplot as plt

network, class_names, class_colors = dn.load_network("/root/ML-final-assignment/yolov4/monitor.cfg",
                              "/root/ML-final-assignment/yolov4/monitor.data",
                              "/root/ML-final-assignment/yolov4/monitor.weights")
network_width = dn.network_width(network)
network_height = dn.network_height(network)

def image_detector(img, n_width, n_height):
    '''
    使用yolo辨識圖片
    '''
    #建立一張Darknet的空白圖片
    darknet_image = dn.make_image(n_width, n_height, 3)
    #將原始圖片轉為RGB格式
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #依據網路的規格調整圖片尺寸 resize(被修改影像, (w, h), interpolation=cv.INTER_LINEAR(插值方式))
    img_resized = cv.resize(img_rgb, (n_width, n_height), interpolation=cv.INTER_LINEAR) 
    # 取得圖片的長寬，讓畫框時比例正常
    img_height, img_width, _ = img.shape
    height_ratio = img_height / n_height
    width_ratio = img_width / n_width
    #將調整過大小的影像插入Darknet空白圖片
    dn.copy_image_from_bytes(darknet_image, img_resized.tobytes())
    #辨識圖片
    detections = dn.detect_image(network, class_names, darknet_image)
    #清除圖片
    dn.free_image(darknet_image)
    return detections, width_ratio, height_ratio

def crop_picture(image, bbox):
    '''
    裁切圖片
    '''
    left, top, right, bottom = dn.bbox2points(bbox)
    left, top, right, bottom = int(left * width_ratio), int(
        top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
    image = image[top:bottom, left:right]
    return image

dir = '/root/ML-final-assignment/yolov4/stove_switch/'
fileList =[f for f in os.listdir(dir) if f.endswith('.jpg')]

fileNum = 0

for fileName in fileList:
    image = cv.imread(dir + fileName)
    detections, width_ratio, height_ratio = image_detector(image, network_width, network_height)
    for label, confidence, bbox in detections:
        img = image.copy()
        img = crop_picture(img, bbox)
        fileNum += 1
        name = "/root/ML-final-assignment/pre_CNN_images/switch_img/switch_%04d.jpg" % fileNum
        cv.imwrite(name, img)
        print(name, end="\r")
print("count: " + str(fileNum))