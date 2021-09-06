from PIL import Image, ImageDraw
import face_recognition          
import cv2
import os
import numpy as np
import torch

path="./facedatalib"
piture=os.listdir(path)
known_face_encodings=[]                                                 #人脸图片数据库
known_face_names=[]

for i in piture:
    if i.endswith(".jpg"):
        known_face_names.append(i)
        known_face_names=[str(j).replace(".jpg","") for j in known_face_names]

    pp=os.path.join(path,i)
    image=face_recognition.load_image_file(pp)
    face_feature=face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_feature)

video_capture = cv2.VideoCapture(0)                   #从opencv获取视频流

process_this_frame = True


while True:
    face_locations = []
    face_encodings = []
    face_names = []
    frames = []
    ret,frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)  #调整视频帧率为真实的1/4，便于快速的识别
    rgb_small_frame = small_frame[:, :, ::-1]                  #将图像从BGR颜色(OpenCV使用)转换为RGB颜色(face_recognition使用)
    
 ##############################为了节省时间，只处理每一帧视频#####################################
    if process_this_frame:
        frames.append(rgb_small_frame)
        if process_this_frame == True:
            batch_of_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0)

            for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):
                number_of_faces_in_frame = len(face_locations)

        #face_locations = face_recognition.face_locations(rgb_small_frame)                  #返回图像中所有人脸边框的numpy数组
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)  #给定一个图像，返回图像中每个人脸的128维人脸编码
        
        face_names = []          
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,0.5)  #人脸特征匹配#查找当前视频帧中所有的人脸和人脸编码
            name = "Unknown"

           #if True in matches:
           #first_match_index = matches.index(True)
           #name = known_face_names[first_match_index]                                 #官方演示代码，人脸抓框比较模糊，这里不采用

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)   #算法提供了一种面孔接近距离比例系数，按照阈值设定比较可以更加的精确
            best_match_index = np.argmin(face_distances)                                           #求出最小的距离值对应的numpy数组底号

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame                                            #一帧数据处理完了，复位标志

  ######################################显示最终结果############################################         
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        
        #top *=4
        #right *=4
        #bottom *=4
        #left *=4                                                                                      #还原图像真实大小

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)                               
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)                  #在人脸框下显示出对应的名字
    cv2.imshow('Video', frame)                                                                           #显示最后结果图像
    if cv2.waitKey(1) & 0xFF == ord('q'):                                                            #按下q键退出，这里也可以用定时器
        break

video_capture.release()
cv2.destroyAllWindows()                                                                           #关闭影像


