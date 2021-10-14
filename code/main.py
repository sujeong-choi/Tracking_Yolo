import time
import cv2
import numpy as np

#from dbr import *
from tracker import *

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]


# def main():
#     filename = "video/test (1).mp4"
#     cap = cv2.VideoCapture(filename)
#     ret, frame = cap.read()
#     height1, width1, _ = frame.shape
#     frame = cv2.resize(frame,(416,416),fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)
#     height2, width2, _ = frame.shape
#     x1 = (130 * 416)//width1
#     x2 = (370 * 416)// width1
#     x3 = 416
#     y1 = (300*416)//height1
#     y2 = (100*416)//height1
#     y3 =(340*416)//height1
#     line = [x1,x2,x3,y1,y2,y3]
#     x1, x2, x3, y1, y2, y3= line
#     frame = cv2.line(frame,(0,416),(x1,y2),(0,255,0),5)
#     frame = cv2.line(frame,(x1,y2),(x2,y2),(0,255,0),5)
#     frame = cv2.line(frame,(x2,y2),(x3,416),(0,255,0),5)
#     # f1 = -(340//110)*cx + 360
#     # f2 = 20
#     # f3 = -(310//(width-380))*cx + 20 - (310*380)//(width-380)
#     a = (y2-y1)//x1
#     b = y1
#     c = y2
#     d = (y3-y2)//(x3-x2)
#     f = y2 - ((y3-y2)*x2)//(x3-x2)
    
#     centers = [[10,20],[250,0],[400,10],[250,200]]
#     for center in centers:
#         cx , cy = center[0], center[1]
#         frame = cv2.line(frame,(cx,cy),(cx,cy),(0,0,255),5)
#         if cx < x1 and cy < a*cx+b: print(f"나갔다 1")
#         elif cx > x1 and cx < x2 and cy < c: print(f"나갔다 2")                
#         elif cx > x2 and cx < x3 and cy < d*cx+f: print(f"나갔다 3")
#         else: print("라인 안에 있다. ")    

#     cv2.imshow("line",frame)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



# def main():
#     images = []
#     count = 0

#     # Create tracker object
    

#     # Initialize Dynamsoft Barcode Reader
#     # reader = BarcodeReader()
#     # Apply for a trial license: https://www.dynamsoft.com/customer/license/trialLicense
#     # license_key = "t0069fQAAAEa2EJJHDv9UNOS3CgvAm98JE+aUy21n+Nw5oho6HLqNBfvxdd6Y0pkdxoVhxLkflSAC3/MHNHziWZZu0LpCWwB4"
#     # reader.init_license(license_key)
#     # decode_data = []

#     # def decodeframe(frame, left, top, right, bottom):
#     #     settings = reader.reset_runtime_settings() 
#     #     settings = reader.get_runtime_settings()
#     #     settings.region_bottom  = bottom
#     #     settings.region_left    = left
#     #     settings.region_right   = right
#     #     settings.region_top     = top
#     #     #settings.barcode_format_ids = EnumBarcodeFormat.BF_QR_CODE
#     #     #settings.expected_barcodes_count = 1
#     #     reader.update_runtime_settings(settings) #지정된 JSON 파일의 설정으로 런타임 설정 update

#     #     try:
#     #         text_results = reader.decode_buffer(frame) # 정의 된 형식의 이미지 픽셀을 포함하는 메모리 버퍼에서 바코드를 디코딩
#     #         if text_results != None:
#     #             for text_result in text_results:
#     #                 decode_data.append(text_result.barcode_text)
#     #         return text_results # 있으면 결과
#     #     except BarcodeReaderError as bre:
#     #         print(bre) # 예외처리

#     #     return None

#     # load classes name
#     # class_names = [barcode, product]
#     #3class_names = []
#     # with open("code.names", "r") as f:
#     #    class_names = [cname.strip() for cname in f.readlines()]

#     # load weights, cfg file
#     net = cv2.dnn.readNet("yolov4-obj_last.weights", "yolov4-obj.cfg")
#     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#     # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
#     # make darknet model  
#     model = cv2.dnn_DetectionModel(net)
#     model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

#     for i in range(1,9):
#         tracker = EuclideanDistTracker()
#         images = []
#         filename = f"video/test ({str(i)}).mp4"
        
#         cap = cv2.VideoCapture(filename)
        
#         print(f"{i}번째 while문 시작")
#         while True:
#             ret, frame = cap.read()
#             if not ret:             #ret이 False면 중지
#                 break
#             frame = cv2.resize(frame,(416,416),fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)
#             height, width, _ = frame.shape
#             # 1. detect object and return output
#             classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
                
#             # 2. Object Tracking
#             # boxes_cids = [x, y, w, h, (class, id)]
#             boxes_cids = tracker.update(classes, boxes, line)
#             for box_cid in boxes_cids:
#                 x, y, w, h, class_id = box_cid
#                 id_name = "BARCODE" if class_id[0] == 0 else "PRODUCT"
#                 text = f"ID ={str(class_id[1])} {id_name}"
#                 cv2.putText(frame, text, (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                

#                 # if class_id[0] == 0:
#                 #     result = decodeframe(frame, x, y, x + w, y + h)
#                 #     if result is None:
#                 #         print("don't decoding")
#                 #         text += f"\n fail decoding"
#                 #     else:
#                 #         print(result[0].barcode_text)
#                 #         text += f"\n success decoding" 
#             cv2.imshow("Frame", frame)
#             images.append(frame)
#             key = cv2.waitKey(30)

#             if key == 27:
#                 image_name = f"output_image/output_image{str(count)}.png"
#                 cv2.imwrite(frame,image_name)

#         print("while 문 완료 비디오 생성")
#         # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#         # output_filename = "output1/out_test("+str(i)+").avi"
#         # out = cv2.VideoWriter(output_filename,fourcc,30.0,(416,416))
#         # for i in range(len(images)):
#         #     out.write(images[i])  

#         #out.release()      
#         cap.release()
#         cv2.destroyAllWindows()


def main():
    images = []
    count = 0
    tracker = EuclideanDistTracker()

    # load weights, cfg file
    net = cv2.dnn.readNet("yolov4-obj_last.weights", "yolov4-obj.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    # make darknet model  
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


    filename = f"../video/test (10).mp4"
    
    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()
    height1, width1, _ = frame.shape

    # x1 = (110 * 416)//width1
    # x2 = (370 * 416)// width1
    # x3 = 416
    # y1 = (300*416)//height1
    # y2 = (20*416)//height1
    # y3 =(340*416)//height1


    x1 = (130 * 416)//width1
    x2 = (370 * 416)// width1
    x3 = 416
    y1 = 416
    y2 = (100*416)//height1
    y3 =416

    line = [x1,x2,x3,y1,y2,y3]

    
    while True:
        if not ret:             #ret이 False면 중지
            break
        frame = cv2.resize(frame,(416,416),fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)
        # 1. detect object and return output
        classes, scores, boxes = model.detect(
        frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        # 2. Object Tracking
        # boxes_cids = [x, y, w, h, (class, id)]
        boxes_cids = tracker.update(classes, boxes, frame, line)
        barcode_data = []
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
            cv2.putText(frame, text, (x-15, y - 15),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.line(frame, (cx, cy), (cx, cy), (0, 0, 255), 5)
            frame = cv2.line(frame, (0, y1), (x1, y2), (255, 255, 0), 3)
            frame = cv2.line(frame, (x1, y2), (x2, y2), (255, 255, 0), 3)
            frame = cv2.line(frame, (x2, y2), (x3, y3), (255, 255, 0), 3)
        
        print(barcode_data)
        # # 1. detect object and return output
        # classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            
        # # 2. Object Tracking
        # # boxes_cids = [x, y, w, h, (class, id)]
        # boxes_cids = tracker.update(classes, boxes, line)
        # for box_cid in boxes_cids:
        #     x, y, w, h, class_id, in_out = box_cid # ^^
        #     cx, cy = (x+x+w)//2, (y+y+h)//2
        #     id_name = "BARCODE" if class_id[0] == 0 else "PRODUCT"
        #     #in_out_string = "IN" if in_out == True else "OUT"
        #     text = f"ID ={str(class_id[1])} {id_name}"
        #     cv2.putText(frame, text, (x-15,y- 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #     if in_out : 
        #         cv2.line(frame,(cx,cy),(cx,cy),(0,0,255),5)
        #     else:
        #         cv2.line(frame,(cx,cy),(cx,cy),(0,0,0),5)
        
        #     #cv2.line(frame,(cx,cy),(cx,cy),(0,0,255),5)
        #     frame = cv2.line(frame,(0,y1),(x1,y2),(255,255,0),3)
        #     frame = cv2.line(frame,(x1,y2),(x2,y2),(255,255,0),3)
        #     frame = cv2.line(frame,(x2,y2),(x3,y3),(255,255,0),3)
            
        cv2.imshow("Frame", frame)
        images.append(frame)
        key = cv2.waitKey(30)
        if key == 27:
            break
        ret, frame = cap.read()

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    output_filename = "output1/out_test(14).avi"
    out = cv2.VideoWriter(output_filename,fourcc,30.0,(416,416))
    for i in range(len(images)):
        out.write(images[i])  
    out.release()
    
    cap.release()
    cv2.destroyAllWindows()


# def main():
#     images = []
#     count = 0
#     line =[]
#     first = True

#     # load weights, cfg file
#     net = cv2.dnn.readNet("yolov4-obj_last.weights", "yolov4-obj.cfg")
#     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#     # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
#     # make darknet model  
#     model = cv2.dnn_DetectionModel(net)
#     model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

#     for i in range(1,17):
#         tracker = EuclideanDistTracker()
#         images = []
#         filename = f"video/test ({str(i)}).mp4"
        
#         cap = cv2.VideoCapture(filename)
        
#         print(f"{i}번째 while문 시작")
#         while True:
#             ret, frame = cap.read()
            
#             if not ret:             #ret이 False면 중지
#                 break

#             if first:
#                 height1, width1, _ = frame.shape
#                 x1 = (110 * 416)//width1
#                 x2 = (370 * 416)// width1
#                 x3 = 416
#                 y1 = (320*416)//height1
#                 y2 = (20*416)//height1
#                 y3 =(340*416)//height1
#                 line = [x1,x2,x3,y1,y2,y3]
#                 first = False

#             frame = cv2.resize(frame,(416,416),fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)
            

#             # 1. detect object and return output
#             classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
                
#             # 2. Object Tracking
#             # boxes_cids = [x, y, w, h, (class, id)]
#             boxes_cids = tracker.update(classes, boxes, line)
#             for box_cid in boxes_cids:
#                 x, y, w, h, class_id, in_out = box_cid # ^^
#                 cx, cy = (x+x+w)//2, (y+y+h)//2
#                 id_name = "BARCODE" if class_id[0] == 0 else "PRODUCT"
#                 text = f"ID ={str(class_id[1])} {id_name}"
#                 frame = cv2.line(frame,(0,y1),(x1,y2),(255,255,0),3)
#                 frame = cv2.line(frame,(x1,y2),(x2,y2),(255,255,0),3)
#                 frame = cv2.line(frame,(x2,y2),(x3,y3),(255,255,0),3)
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
#                 cv2.putText(frame, text, (x-15,y- 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 200), 1)
#                 if in_out : 
#                     cv2.line(frame,(cx,cy),(cx,cy),(0,0,255),5)
#                 else:
#                     cv2.line(frame,(cx,cy),(cx,cy),(0,0,0),5)

#                 cv2.imshow("Frame", frame)
#                 images.append(frame)
#                 key = cv2.waitKey(30)

#                 if key == 27:
#                     break

#         print("while 문 완료 비디오 생성")
#         fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#         output_filename = "output2/out_test("+str(i)+").avi"
#         out = cv2.VideoWriter(output_filename,fourcc,30.0,(416,416))
#         for i in range(len(images)):
#             out.write(images[i])  

#         out.release()      
#         cap.release()
#         cv2.destroyAllWindows()


if __name__ == '__main__':
    # in_out = False
    # dic = {(0,1):(1,2,3,4,in_out)}
    # print(dic)
    main()
