import cv2
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from sort import Sort
import os

os.makedirs('final_frames', exist_ok=True)

BLUE_LINE = [(283,470), (615,470)]
GREEN_LINE = [(210,500),  (634,500)]
RED_LINE = [(203,530), (639,530)]

cross_blue_line = {}
cross_green_line = {}
cross_red_line = {}

avg_speeds = {}

FACTOR_KM = 3.6
LATENCY_FPS = 15

def euclidean_distance(point1: tuple, point2: tuple):
    x1, y1 = point1
    x2, y2 = point2
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distance

def calculate_avg_speed(track_id):
    time_bg = (cross_green_line[track_id]['time'] - cross_blue_line[track_id]['time']).total_seconds()
    time_gr = (cross_red_line[track_id]['time'] - cross_green_line[track_id]['time']).total_seconds()

    distance_bg = euclidean_distance(cross_green_line[track_id]['point'], cross_blue_line[track_id]['point'])
    distance_gr = euclidean_distance(cross_red_line[track_id]['point'], cross_green_line[track_id]['point'])

    if time_bg > 0 and time_gr > 0:
        speed_bg = round((distance_bg / (time_bg * VIDEO_FPS)) * (FACTOR_KM * LATENCY_FPS), 2)
        speed_gr = round((distance_gr / (time_gr * VIDEO_FPS)) * (FACTOR_KM * LATENCY_FPS), 2)
        return round((speed_bg + speed_gr) / 2, 2)
    return 0

if __name__ == '__main__':
    cap = cv2.VideoCapture('velocidad.mp4')
    VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS)

    model = YOLO('yolo11n.pt')
    tracker = Sort()

    frame_id = 0

    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            break

        frame = cv2.resize(frame, (1280, 720))
        height, width, _ = frame.shape
        frame_cropped = frame[:, :int(width * 0.6)]

        results = model(frame_cropped, stream=True)

        for res in results:
            filtered_indices = np.where((np.isin(res.boxes.cls.cpu().numpy(), [2, 3, 5, 7])) & (res.boxes.conf.cpu().numpy() > 0.3))[0]
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)

            tracks = tracker.update(boxes).astype(int)

            for xmin, ymin, xmax, ymax, track_id in tracks:
                xc, yc = int((xmin + xmax) / 2), ymax

                if track_id not in cross_blue_line:
                    cross_blue = (BLUE_LINE[1][0] - BLUE_LINE[0][0]) * (yc - BLUE_LINE[0][1]) - (BLUE_LINE[1][1] - BLUE_LINE[0][1]) * (xc - BLUE_LINE[0][0])
                    if cross_blue >= 0:
                        cross_blue_line[track_id] = {'time': datetime.now(), 'point': (xc, yc)}

                elif track_id not in cross_green_line and track_id in cross_blue_line:
                    cross_green = (GREEN_LINE[1][0] - GREEN_LINE[0][0]) * (yc - GREEN_LINE[0][1]) - (GREEN_LINE[1][1] - GREEN_LINE[0][1]) * (xc - GREEN_LINE[0][0])
                    if cross_green >= 0:
                        cross_green_line[track_id] = {'time': datetime.now(), 'point': (xc, yc)}

                elif track_id not in cross_red_line and track_id in cross_green_line:
                    cross_red = (RED_LINE[1][0] - RED_LINE[0][0]) * (yc - RED_LINE[0][1]) - (RED_LINE[1][1] - RED_LINE[0][1]) * (xc - RED_LINE[0][0])
                    if cross_red >= 0:
                        cross_red_line[track_id] = {'time': datetime.now(), 'point': (xc, yc)}
                        avg_speed = calculate_avg_speed(track_id)
                        avg_speeds[track_id] = f'{avg_speed} Km/h'

                
                if track_id in avg_speeds:
                    cv2.putText(frame, avg_speeds[track_id], (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

                # cv2.circle(frame, (xc, yc), 5, (0, 255, 0), 1)
                # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)

        cv2.line(frame_cropped, BLUE_LINE[0], BLUE_LINE[1], (255, 0, 0), 3)
        cv2.line(frame_cropped, GREEN_LINE[0], GREEN_LINE[1], (0, 255, 0), 3)
        cv2.line(frame_cropped, RED_LINE[0], RED_LINE[1], (0, 0, 255), 3)

        frame_path = f'final_frames/frame_{frame_id:05d}.jpg'
        cv2.imwrite(frame_path, frame)
        frame_id += 1

        cv2.imshow('frame', frame_cropped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
