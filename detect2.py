import argparse
import time
from pathlib import Path

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


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

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

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    
                    #inspection area fill for box
                    areaFillStatus = inspectAreaFill(cls=cls, classNumber=0, xyxy=xyxy, pointLocation=pointLocation, areaFillStatus=areaFillStatus)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            
            # Draw fill box for each area on image im0
            im0= drawFillBox(im0=im0, line_thickness=2, pointLocation=pointLocation, areaFillStatus=areaFillStatus, marginAreaFill=5)            
            print(areaFillStatus)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
