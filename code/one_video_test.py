import time
import cv2
import numpy as np

#from dbr import *
from tracker_with_area import *

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]


def main():
    images = []
    count = 0
    line =[]
    first = True
    

    # load weights, cfg file
    net = cv2.dnn.readNet("yolov4-obj_last.weights", "yolov4-obj.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    # make darknet model  
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

    

    tracker = EuclideanDistTracker()
    images = []
    filename = "data/tracking/test/test (2).mp4"
    
    cap = cv2.VideoCapture(filename)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:             #ret이 False면 중지
            break
        barcode_data = []

        # if first:
        #     height1, width1, _ = frame.shape
        #     x1 = (110 * 416)//width1
        #     x2 = (370 * 416)// width1
        #     x3 = 416
        #     y1 = (320*416)//height1
        #     y2 = (20*416)//height1
        #     y3 =(340*416)//height1
        #     line = [x1,x2,x3,y1,y2,y3]
        #     first = False
        height1, width1, _ = frame.shape
        x1 = (110 * 416)//width1
        x2 = (370 * 416)// width1
        x3 = 416
        y1 = (320*416)//height1
        y2 = (20*416)//height1
        y3 =(340*416)//height1
        line = [x1,x2,x3,y1,y2,y3]

        origin = frame
        frame = cv2.resize(frame,(416,416),fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)

        # 1. detect object and return output
        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            
        # 2. Object Tracking
        # boxes_cids = [x, y, w, h, (class, id)]
        boxes_cids = tracker.update(classes, boxes, frame,line)
        for box_cid in boxes_cids:
            x, y, w, h, class_id, in_out, decode_info = box_cid
            cx, cy = (x+x+w)//2, (y+y+h)//2
            if class_id[0] == 0:
                id_name = "BARCODE " + decode_info
                barcode_data.append(decode_info)
            else:
                id_name = "PRODUCT"
            in_out_string = "IN" if in_out == True else "OUT"
            text = f"{in_out_string} ID ={str(class_id[1])} {id_name}"
            cv2.putText(frame, text, (x-15, h+y- 15),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.line(frame, (cx, cy), (cx, cy), (0, 0, 255), 5)
            if in_out : 
                cv2.line(frame,(cx,cy),(cx,cy),(0,0,255),5)
            else:
                cv2.line(frame,(cx,cy),(cx,cy),(0,0,0),5)
            #frame = cv2.line(frame, (0, y1), (x1, y2), (255, 255, 0), 3)
            #frame = cv2.line(frame, (x1, y2), (x2, y2), (255, 255, 0), 3)
            #frame = cv2.line(frame, (x2, y2), (x3, y3), (255, 255, 0), 3)

            cv2.imshow("Frame", frame)
            images.append(frame)
            key = cv2.waitKey(30)

            if key == 27:
                break

    print("while 문 완료 비디오 생성")
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    output_filename = "data/tracking/output_with_area/test/test (2)(2).avi"
    out = cv2.VideoWriter(output_filename,fourcc,30.0,(416,416))
    for i in range(len(images)):
        out.write(images[i])  

    out.release()      
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # in_out = False
    # dic = {(0,1):(1,2,3,4,in_out)}
    # print(dic)
    main()
