# 機器學習 期末作業

###### tags: `Ml`, `YOLO`, `CNN`

資管三B 10814032 葉哲丞

# 研究目的

判斷出瓦斯爐的開關狀態，讓程式能進行下一階段的判斷。

# 使用資料集來源

與專題組員共同至IKEA，拍攝瓦斯爐各種狀態的照片。

# 使用方法


# 實驗設計

# 實驗結果與討論

# 訓練準備指令

cp -r s10814032/final/* ./

wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29

cp darknet/cfg/yolov4-tiny.cfg ./model/

cd darknet/

time darknet detector train /root/model/monitor.data /root/model/monitor-tiny.cfg yolov4.conv.137 -dont_show -map
