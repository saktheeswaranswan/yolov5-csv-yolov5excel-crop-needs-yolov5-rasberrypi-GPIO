import argparse
import csv
import datetime
import os
import platform
import sys
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import (check_imshow, check_img_size, increment_path, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_txt=False,  # save results to *.txt
        classes=None,  # filter by class: --classes 0, or --classes 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        update=False,  # update all models
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or (is_url and not is_file)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    else:
        raise ValueError("Invalid source type. Only webcam source is supported for live video detection.")

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    windows, dt = [], (Profile(), Profile(), Profile())
    csv_file = None

    for path, im, im0s, _, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        with dt[1]:
            pred = model(im, augment=augment, profile=dt[2], visualize=False)[0]

        with dt[2]:
            if platform.system() != 'Darwin':  # macOS
                model.float()  # for model profiler

            # Process predictions
            detections = []
            for i, det in enumerate(pred):  # per image
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
                det[:, :2] -= det[:, 2:] / 2  # xy center to top-left corner
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        label = f'{names[int(cls)]} {conf:.2f}'
                        save_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=line_thickness)

                    # Add detection to list
                    detections.append((names[int(cls)], conf, *xyxy))

                # Log detections to CSV file
                if csv_file is None:
                    csv_file = create_csv_file()

                log_detections(csv_file, detections)

        if platform.system() != 'Darwin':  # macOS
            torch.cuda.empty_cache() if device.type != 'cpu' else None

        # Stream results
        dt_msg = ' '.join(f'{x:.2f}' + s for x, s in zip(dt.value, ('FPS', 'classification', 'NMS')))
        plot_pred = False
        if len(windows) == 1:
            plot_pred = True
            p = windows.pop()
        elif len(windows) == 0:
            plot_pred = True

        if plot_pred:
            p = Annotator(im0s, line_thickness=line_thickness, hide_labels=hide_labels, hide_conf=hide_conf)
            windows.append(p)

        if len(p) == 1:
            im0s = im0s.transpose(1, 2, 0)[:, :, ::-1]  # BGR to RGB
            if p.mode == 'image':
                p.image(im0s)
            elif p.mode == 'save':
                p.save(im0s, path)

        # Print results
        for i, det in enumerate(pred):  # detections per image
            p.prediction(det, names, save_dir=None, show=True, line_thickness=line_thickness)

        # Stream results
        if len(windows) == 1:
            p.stream(dt_msg)
        elif len(windows) == 0:
            sys.exit()

    # Close CSV file if opened
    if csv_file is not None:
        csv_file.close()


def create_csv_file():
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_path = increment_path('detections', ext='.csv')
    csv_file = open(file_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['Timestamp', 'Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])
    return csv_file


def log_detections(csv_file, detections):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    writer = csv.writer(csv_file)
    for detection in detections:
        row = [timestamp] + list(detection)
        writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='detect.py')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/dir/URL/glob, 0 for webcam
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--line-thickness', type=int, default=3, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()

    ROOT = Path(__file__).resolve().parent
    sys.path.append(str(ROOT))  # add yolov5/ to path

    run(**vars(opt))

