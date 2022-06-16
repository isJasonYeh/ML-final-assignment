import random
import os

probability = 0.8
#train資料佔的比例，valid為1-probability

dir = '/root/stove_switch/'

fileList =[f for f in os.listdir(dir) if f.endswith('.jpg')]
#使用os.listdir取得該目錄下的所有檔案名稱，並且用endswith()判斷後面的字(附檔名)，為True的才存到Tuple裡面

random.shuffle(fileList) #打亂順序

with open('/root/Final/yolov4/train.txt', 'w') as train, open('/root/Final/yolov4/train.txt', 'w') as valid:
  for fileName in fileList:
    if (random.random() <= probability):
      train.write(dir + fileName + "\n")
    else:
      valid.write(dir + fileName + "\n")