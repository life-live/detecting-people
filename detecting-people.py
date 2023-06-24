import os
import time

import cv2
import numpy
from pygrabber.dshow_graph import FilterGraph
from ultralytics import YOLO


class DetectingPeople:
    def __init__(self, model_path: str = "models/yolov8s.pt", video_output_to_the_screen: bool = True,
                 camera_index: int = 0):
        os.environ['CUDA_MODULE_LOADING'] = "LAZY"
        self.graph = FilterGraph()
        self.time_of_the_last_object = None  # Время, когда был виден последний объект
        self.camera_name = self.get_camera_name(camera_index)  # Имя камеры, которая будет использоваться
        self.camera_width = 640  # Ширина разрешения камеры
        self.camera_height = 480  # Высота разрешения камеры
        self.camera_fps = 30  # Поддерживаемое количество кадров в секунду у камеры
        self.video_output_to_the_screen = video_output_to_the_screen  # Выводить ли видео на экран
        self.output_of_all_cameras()  # Вывод всех камер
        self.model = YOLO(model_path, task="detect")  # Использование обученной модели искусственного интеллекта
        self.video_capture = self.initializing_the_camera(camera_index)  # Инициализация камеры
        self.video_recording = self.initializing_video_recording()  # Инициализация видео-съемки

    def output_of_all_cameras(self):
        cameras = self.graph.get_input_devices()
        for camera in cameras:
            camera_index = self.graph.get_input_devices().index(camera)
            print(f"{camera_index}: {camera}")

    def get_camera_name(self, camera_index):
        camera_name = self.graph.get_input_devices()[camera_index]
        return camera_name

    def initializing_the_camera(self, camera_index: int):
        video_capture = cv2.VideoCapture(camera_index)  # Получение видео-потока
        video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))
        self.camera_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # Ширина кадра
        self.camera_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Высота кадра
        self.camera_fps = int(video_capture.get(cv2.CAP_PROP_FPS))  # Кадры в секунду
        return video_capture

    def initializing_video_recording(self):
        if not os.path.exists("videos"):
            os.mkdir("videos")
        date_with_time = time.strftime("%d-%m-%Y %H.%M.%S", time.localtime())  # Текущее время и дата
        video_recording = cv2.VideoWriter(f"videos/{date_with_time}.mkv", cv2.VideoWriter_fourcc(*"H264"),
                                          self.camera_fps,
                                          (self.camera_width, self.camera_height))  # Создание видео-файла для записи
        return video_recording

    def detecting_objects_in_an_image(self, frame: numpy.ndarray, device: int = 0, conf: float = 0.45,
                                      classes: int or list or None = 0):
        results = self.model.predict(source=frame, device=device, conf=conf, int8=True, half=True, classes=classes)
        return results

    def drawing_data_on_an_image(self, results: list):
        # Обвод объектов
        annotated_frame = results[0].plot()

        # Вставляем текущее время на кадр
        frame_time = time.strftime("%H:%M:%S", time.localtime())
        cv2.putText(annotated_frame, frame_time, (5, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 0, 0), 9)
        cv2.putText(annotated_frame, frame_time, (5, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255, 255), 2)

        # Вставляем текущую дату на кадр
        frame_date = time.strftime("%d/%m/%Y", time.localtime())
        cv2.putText(annotated_frame, frame_date, (self.camera_width - 160, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 0, 0), 9)
        cv2.putText(annotated_frame, frame_date, (self.camera_width - 160, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255, 255), 2)
        return annotated_frame

    def checking_objects_and_video_recording(self, frame: numpy.ndarray, results: list):
        current_objects = [result.names[int(cls)] for result in results for cls in result.boxes.cls]
        if "person" in current_objects:
            self.video_recording.write(frame)
            self.time_of_the_last_object = time.time()
        else:
            # Если прошло менее 3 секунд с момента последнего обнаружения объекта, записываем кадр
            if self.time_of_the_last_object is not None and time.time() - self.time_of_the_last_object <= 3:
                self.video_recording.write(frame)
            # Если прошло более 3 секунд с момента последнего обнаружения объекта, сбрасываем время
            else:
                self.time_of_the_last_object = 0

    def releasing_resources(self):
        self.video_capture.release()
        self.video_recording.release()
        cv2.destroyAllWindows()

    def start(self):
        print("DetectingPeople started...")
        if self.video_output_to_the_screen:
            cv2.namedWindow(self.camera_name, cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(self.camera_name, self.camera_width, self.camera_height)
        while True:
            _, frame = self.video_capture.read()
            if frame is None:
                continue
            result = self.detecting_objects_in_an_image(frame=frame)
            annotated_frame = self.drawing_data_on_an_image(result)
            self.checking_objects_and_video_recording(annotated_frame, result)
            if self.video_output_to_the_screen:
                cv2.imshow(self.camera_name, annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.releasing_resources()


if __name__ == "__main__":
    dp = DetectingPeople()
    dp.start()
