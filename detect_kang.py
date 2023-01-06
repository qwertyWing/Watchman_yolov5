import argparse
import os
import sys
import time
from pathlib import Path
import sound

import threading_ex

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import LED_zn
import threading
import datetime

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (check_img_size, check_requirements, cv2,
                           non_max_suppression, print_args)
from utils.torch_utils import select_device
from utils.augmentations import letterbox

import cvcamera as ca

@torch.no_grad()


def Save():
	ROOT_PATH = "/media/nvidia"
	disk = os.listdir(ROOT_PATH)
	for d in disk :
		try : 
			os.listdir(os.path.join(ROOT_PATH, d))
		except OSError as e :
			os.system(f"sudo umount {os.path.join(ROOT_PATH, d)}")
			os.system(f"sudo rm -rf {os.path.join(ROOT_PATH, d)}")
		except Exception as e:
			pass
		else :
			return d
	return None


def GetVideoOut(c_usb,file_name,width, height) :
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    return cv2.VideoWriter(f"{os.path.join('./video',c_usb,file_name)}",fourcc,15.0,(width,height),)

def GetOutputFileName() :
    file_name = ""
    current_date = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    file_name = f"{current_date}.avi"
    return file_name


def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/trash.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    # Load camera
    
    current_usb = Save()
    #cam_ids = [0]  # ca.mutlicamera()
    #video_captures = [cv2.VideoCapture("rtsp://192.168.0.99/H.264/media.smp") for idx in cam_ids]
    video_captures = [cv2.VideoCapture("rtsp://192.168.0.99/H.264/media.smp")]
    
    fps = 15.0
    bs = len(video_captures)
    
    print(f"{bs}cameras")
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    cudnn.benchmark = True  # set True to speed up constant image size inference
    vid_start = 0
    video = [None] * bs
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    count = 0
    number = 0
    save_video = False
    # Video setting
    o = 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    while True:
        # Get image
        frames = ca.get_image(video_captures)

        if frames in [None]:
            continue

        # 이미지 전처리 시작
        # path, im, im0s, vid_cap => None, img, frames, None
        im = frames.copy()
        im = [letterbox(x, imgsz, stride=stride, auto=True)[0] for x in im]

        # Stack
        im = np.stack(im, 0)

        # Convert
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        im = np.ascontiguousarray(im)
        # 이미지 전처리 끝

        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()
        im /= 255

        # Inference
        pred = model(im, augment=augment, visualize=visualize)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        anno_frames = []
        for i, det in enumerate(pred):
            print(f"Process camera {i}")
            anno_frames.append(frames[i].copy())
            if len(det):
                # 인식된 객체가 있는 경우
                for *xyxy, conf, cls in reversed(det):
                    lx, ly, rx, ry = list(map(int,xyxy))
                    if int(cls) ==0 and float(conf) > 0.4:
                        cv2.rectangle(anno_frames[i], (lx, ly), (rx, ry), (255, 255, 0), 0, cv2.LINE_AA)
                        cv2.putText(anno_frames[i], names[int(cls)], (lx, ly + 20), 0, 1, (255, 255, 0), 1, cv2.LINE_AA)
                        #print(xyxy, conf, cls)
                        #print(names[int(cls)]) #인식된 이름 값 불러오기

                        

                        if video[i] is None:
                            threading_ex.Threadingstart()
                            vid_start = time.time()
                            save_video = True
                            for idx in range(bs) :
                                #video[idx] = cv2.VideoWriter('./video' + str(idx) + '{}.mp4'.format(o), fourcc,fps, (640, 640))
                                video[idx] = cv2.VideoWriter(f'/media/nvidia/D2E8A466E8A44B15/video_{number}.mp4'.format(o), fourcc,fps, (640, 480))
                                #video[idx] = cv2.VideoWriter(f'./video/video_{number}.mp4'.format(o), fourcc,fps, (640, 480))
                                number += 1                                
            else:
                # 인식된 객체가 없는 경우
                pass
            # 인식 됐거나 인식 안 된 경우ph
            if save_video:
                video[i].write(frames[i])
                print("\n.....................................Recording is processing")
                if time.time() - vid_start >= 20:
                    save_video = False
                    for idx in range(bs):
                        video[idx].release()
                        video[idx] = None
                    print("\n.....................................End of recording")
        

        for idx, f in enumerate(anno_frames):
            #cv2.imwrite(f"./images/cam{idx}_{count}.jpg", f)
            cv2.imshow(f"f{idx}", f)
            count += 1
        cv2.waitKey(1)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'stream.txt', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/trash.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', default=True, action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
