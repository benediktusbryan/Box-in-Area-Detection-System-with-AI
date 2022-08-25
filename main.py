import fastapi
import uvicorn
import cv2
import imutils
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, Response as HTTPResponse
from imutils.video import VideoStream
from fastapi.middleware.cors import CORSMiddleware
import os
import datetime
import argparse
import time
from pathlib import Path
from fastapi_utils.tasks import repeat_every
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.datasets import letterbox
import numpy as np

class FastAPP(FastAPI):
    video_stream: VideoStream
    ai_ready = False
    frame = None
    inferenceReady = False

app = FastAPP()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.video_stream = VideoStream('rtsp://air:zxcasdqwe123@172.16.100.222:554/Streaming/channels/1')

@app.on_event('startup')
def startup():
    print("Starting Video Stream!")
    app.video_stream.start()
    print("Done Startup!")

@app.on_event('shutdown')
def shutdown():
    print("Shutdown Video Stream")
    app.video_stream.stop()
    print("Done Shutdown!")

def stream_generator():
    try: 
        while app.video_stream.stream.grabbed:
            # frame = app.video_stream.read() if app.frame == None else app.frame
            frame = app.frame
            # frame = imutils.resize(frame, 680, 480)
            if frame is None: continue
            ret, img = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(img) + b'\r\n')
    except Exception as e: print("[ERROR] ", e)


@app.get('/hasil')
def stream_mjpg():
    return app.data_inference
@app.get('/mjpg')
async def stream_mjpg():
    return StreamingResponse(stream_generator(), media_type="multipart/x-mixed-replace;boundary=frame" )
@app.get('/available_record')
async def get_available_date():
    list_directory = os.listdir('//venus.airlab.id/Public/CCTV')
    list_directory = [d for d in list_directory if '.' not in d]
    available_date = [datetime.datetime.strptime(d, '%YY%mM%dD%HH') for d in list_directory]
    ret = []
    for directory, date in zip(list_directory, available_date):
        ret.append({'date': date, "directory": directory})
    return ret

def init_ai():
    
    weights, view_img, save_txt, imgsz, trace =  "model1.pt", False, False, 640, not False
    # Initialize
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, imgsz)

    # if half:
    #     model.half()  # to FP16
    # Second-stage classifier
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    app.names = names
    app.colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    app.old_img_w = app.old_img_h = imgsz
    app.old_img_b = 1
    app.model = model
    app.device = device
    print("init done")
    app.ai_ready = True
    app.inferenceReady = True


# @repeat_every(seconds=0.1)
# @app.on_event('startup')
# def call_inference():
#     if app.ai_ready:
#         inference()

@app.on_event('startup')
@repeat_every(seconds=1/5)
def inference():
    try:
        if not app.ai_ready: return
        print("dor")
        # if not app.video_stream.stream.grabbed: return
        # print("dar")
        # if not app.inferenceReady: return
        # print("der")
        app.inferenceReady = False
        print("start inference")
        frame = app.video_stream.read()
        # check for common shapes
        stride = int(app.model.stride.max())
        s = np.stack([letterbox(frame, 640, stride=stride)[0].shape], 0)  # shapes
        rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        # Letterbox
        img = [letterbox(frame, 640, auto=rect, stride=stride)[0]]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(app.device)
        img =  img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        old_img_b = app.old_img_b
        old_img_h = app.old_img_h
        old_img_w = app.old_img_w
        if app.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            print("shape=", img.shape)
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                app.model(img, augment=False)[0]

        # Inference
        pred = app.model(img, augment=False)[0]
        
        app.old_img_b = old_img_b
        app.old_img_h = old_img_h
        app.old_img_w = old_img_w
        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=0, agnostic=False)
        for i, det in enumerate(pred):
            print("Run Inference")
            
            #init dict result variable
            areaFillStatus ={
                                "area 0": "0",
                                "area 1": "0",
                                "area 2": "0",
                                "area 3": "0",
                                "area 4": "0",
                                "area 5": "0",
                                "area 6": "0",
                                "area 7": "0",
                                "area 8": "0",
                            }                
            #calculate point location
            pointLocation = calculateAreaFill(x1=770, y1=590, x2=1225, y2=975, xAxis=3, yAxis=3)

            if len(det):
                print("Det")

                s =""
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {app.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # app.data_inference = s

                # Write results
                for *xyxy, conf, cls in reversed(det):                   
                    #inspection area fill for box
                    areaFillStatus = inspectAreaFill(cls=cls, classNumber=0, xyxy=xyxy, pointLocation=pointLocation, areaFillStatus=areaFillStatus)

                    label = f'{app.names[int(cls)]} {conf:.2f}'
                    # plot_one_box(xyxy, frame, label=label, color=app.colors[int(cls)], line_thickness=1)
                    plot_one_box(xyxy, frame, label=label, color=[0, 0, 0], line_thickness=1)
                    # app.frame = frame
                # app.data_inference = s

            # Draw fill box for each area on image im0
            app.frame = drawFillBox(im0=frame, line_thickness=2, pointLocation=pointLocation, areaFillStatus=areaFillStatus, marginAreaFill=5)            
            app.data_inference = areaFillStatus
            print(areaFillStatus)
        app.inferenceReady = True
        print("Run Inference")
    except Exception as e:
        print("nyangkoot", e)


def calculateAreaFill(x1, y1, x2, y2, xAxis, yAxis):
    xyxyAreaFill = [x1, y1, x2, y2]   #[x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    xywhAreaFill = xyxyAreaFill   #[x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    xywhAreaFill[0] = xyxyAreaFill[0]   #x top-left
    xywhAreaFill[1] = xyxyAreaFill[1]   #y top-left
    xywhAreaFill[2] = xyxyAreaFill[2]-xyxyAreaFill[0]   #width
    xywhAreaFill[3] = xyxyAreaFill[3]-xyxyAreaFill[1]   #height
    areaSize = [xAxis, yAxis]
    pointLocation = []
    for j in range(areaSize[1]+1):
        for i in range(areaSize[0]+1):
            x = xywhAreaFill[2]/areaSize[0]*i + xywhAreaFill[0]
            y = xywhAreaFill[3]/areaSize[1]*j + xywhAreaFill[1]
            pointLocation.append([x,y]) #x1,y1 ; x2,y1 ; x3,y1
    return pointLocation

def inspectAreaFill(cls, classNumber, xyxy, pointLocation, areaFillStatus):
    if cls == classNumber:    #check area fill with box
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
        if pointLocation[0][0] <= xywh[0] <= pointLocation[5][0] and pointLocation[0][1]<=xywh[1]<=pointLocation[5][1]:
            areaFillStatus["area 0"] = "1"
            print("area 0 is filled")
        elif pointLocation[1][0] <= xywh[0] <= pointLocation[6][0] and pointLocation[1][1]<=xywh[1]<=pointLocation[6][1]:
            areaFillStatus["area 1"] = "1"
            print("area 1 is filled")
        elif pointLocation[2][0] <= xywh[0] <= pointLocation[7][0] and pointLocation[2][1]<=xywh[1]<=pointLocation[7][1]:
            areaFillStatus["area 2"] = "1"
            print("area 2 is filled")

        elif pointLocation[4][0] <= xywh[0] <= pointLocation[9][0] and pointLocation[4][1]<=xywh[1]<=pointLocation[9][1]:
            areaFillStatus["area 3"] = "1"
            print("area 3 is filled")
        elif pointLocation[5][0] <= xywh[0] <= pointLocation[10][0] and pointLocation[5][1]<=xywh[1]<=pointLocation[10][1]:
            areaFillStatus["area 4"] = "1"
            print("area 4 is filled")
        elif pointLocation[6][0] <= xywh[0] <= pointLocation[11][0] and pointLocation[6][1]<=xywh[1]<=pointLocation[11][1]:
            areaFillStatus["area 5"] = "1"
            print("area 5 is filled")

        elif pointLocation[8][0] <= xywh[0] <= pointLocation[13][0] and pointLocation[8][1]<=xywh[1]<=pointLocation[13][1]:
            areaFillStatus["area 6"] = "1"
            print("area 6 is filled")
        elif pointLocation[9][0] <= xywh[0] <= pointLocation[14][0] and pointLocation[9][1]<=xywh[1]<=pointLocation[14][1]:
            areaFillStatus["area 7"] = "1"
            print("area 7 is filled")
        elif pointLocation[10][0] <= xywh[0] <= pointLocation[15][0] and pointLocation[10][1]<=xywh[1]<=pointLocation[15][1]:
            areaFillStatus["area 8"] = "1"
            print("area 8 is filled")
    return areaFillStatus
    
def drawFillBox(im0, line_thickness, pointLocation, areaFillStatus, marginAreaFill):
    tl = line_thickness or round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1  # line/font thickness            
    tf = max(tl - 1, 1)  # font thickness
    colorBlue = [255, 0, 0]
    colorRed = [0, 0, 255]

    if areaFillStatus["area 0"] == "1":
        colorArea0 = colorRed
    else:
        colorArea0 = colorBlue
    if areaFillStatus["area 1"] == "1":
        colorArea1 = colorRed
    else:
        colorArea1 = colorBlue
    if areaFillStatus["area 2"] == "1":
        colorArea2 = colorRed
    else:
        colorArea2 = colorBlue
    if areaFillStatus["area 3"] == "1":
        colorArea3 = colorRed
    else:
        colorArea3 = colorBlue
    if areaFillStatus["area 4"] == "1":
        colorArea4 = colorRed
    else:
        colorArea4 = colorBlue
    if areaFillStatus["area 5"] == "1":
        colorArea5 = colorRed
    else:
        colorArea5 = colorBlue
    if areaFillStatus["area 6"] == "1":
        colorArea6 = colorRed
    else:
        colorArea6 = colorBlue
    if areaFillStatus["area 7"] == "1":
        colorArea7 = colorRed
    else:
        colorArea7 = colorBlue
    if areaFillStatus["area 8"] == "1":
        colorArea8 = colorRed
    else:
        colorArea8 = colorBlue

    overlay = im0.copy()
    cv2.rectangle(overlay, (int(pointLocation[0][0]+marginAreaFill), int(pointLocation[0][1]+marginAreaFill)), (int(pointLocation[5][0]-marginAreaFill), int(pointLocation[5][1]-marginAreaFill)), colorArea0, -1)  # filled
    cv2.rectangle(overlay, (int(pointLocation[1][0]+marginAreaFill), int(pointLocation[1][1]+marginAreaFill)), (int(pointLocation[6][0]-marginAreaFill), int(pointLocation[6][1]-marginAreaFill)), colorArea1, -1)  # filled
    cv2.rectangle(overlay, (int(pointLocation[2][0]+marginAreaFill), int(pointLocation[2][1]+marginAreaFill)), (int(pointLocation[7][0]-marginAreaFill), int(pointLocation[7][1]-marginAreaFill)), colorArea2, -1)  # filled            
    cv2.rectangle(overlay, (int(pointLocation[4][0]+marginAreaFill), int(pointLocation[4][1]+marginAreaFill)), (int(pointLocation[9][0]-marginAreaFill), int(pointLocation[9][1]-marginAreaFill)), colorArea3, -1)  # filled
    cv2.rectangle(overlay, (int(pointLocation[5][0]+marginAreaFill), int(pointLocation[5][1]+marginAreaFill)), (int(pointLocation[10][0]-marginAreaFill), int(pointLocation[10][1]-marginAreaFill)), colorArea4, -1)  # filled
    cv2.rectangle(overlay, (int(pointLocation[6][0]+marginAreaFill), int(pointLocation[6][1]+marginAreaFill)), (int(pointLocation[11][0]-marginAreaFill), int(pointLocation[11][1]-marginAreaFill)), colorArea5, -1)  # filled
    cv2.rectangle(overlay, (int(pointLocation[8][0]+marginAreaFill), int(pointLocation[8][1]+marginAreaFill)), (int(pointLocation[13][0]-marginAreaFill), int(pointLocation[13][1]-marginAreaFill)), colorArea6, -1)  # filled
    cv2.rectangle(overlay, (int(pointLocation[9][0]+marginAreaFill), int(pointLocation[9][1]+marginAreaFill)), (int(pointLocation[14][0]-marginAreaFill), int(pointLocation[14][1]-marginAreaFill)), colorArea7, -1)  # filled
    cv2.rectangle(overlay, (int(pointLocation[10][0]+marginAreaFill), int(pointLocation[10][1]+marginAreaFill)), (int(pointLocation[15][0]-marginAreaFill), int(pointLocation[15][1]-marginAreaFill)), colorArea8, -1)  # filled
    
    alpha = 0.3  # Transparency factor.            
    im0 = cv2.addWeighted(overlay, alpha, im0, 1 - alpha, 0)

    cv2.putText(im0, "0", (int(pointLocation[0][0]+marginAreaFill), int(pointLocation[0][1]+marginAreaFill+20)), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    cv2.putText(im0, "1", (int(pointLocation[1][0]+marginAreaFill), int(pointLocation[1][1]+marginAreaFill+20)), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    cv2.putText(im0, "2", (int(pointLocation[2][0]+marginAreaFill), int(pointLocation[2][1]+marginAreaFill+20)), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    cv2.putText(im0, "3", (int(pointLocation[4][0]+marginAreaFill), int(pointLocation[4][1]+marginAreaFill+20)), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    cv2.putText(im0, "4", (int(pointLocation[5][0]+marginAreaFill), int(pointLocation[5][1]+marginAreaFill+20)), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    cv2.putText(im0, "5", (int(pointLocation[6][0]+marginAreaFill), int(pointLocation[6][1]+marginAreaFill+20)), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    cv2.putText(im0, "6", (int(pointLocation[8][0]+marginAreaFill), int(pointLocation[8][1]+marginAreaFill+20)), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    cv2.putText(im0, "7", (int(pointLocation[9][0]+marginAreaFill), int(pointLocation[9][1]+marginAreaFill+20)), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    cv2.putText(im0, "8", (int(pointLocation[10][0]+marginAreaFill), int(pointLocation[10][1]+marginAreaFill+20)), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return im0


if __name__ == '__main__':
    init_ai()
    uvicorn.run(app, host="0.0.0.0")