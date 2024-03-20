import json
import os
import os.path as osp
from multiprocessing import Array

import cv2
import numpy as np

from gesture_recognition_tools.utils.common import load_core
from gesture_recognition_tools.utils.person_detector import PersonDetector
from gesture_recognition_tools.utils.tracker import Tracker
from gesture_recognition_tools.utils.action_recognizer import ActionRecognizer

DETECTOR_OUTPUT_SHAPE = -1, 5
TRACKER_SCORE_THRESHOLD = 0.4
TRACKER_IOU_THRESHOLD = 0.3
ACTION_IMAGE_SCALE = 256
OBJECT_IDS = [ord(str(n)) for n in range(10)]

# Custom Paths
openvinoPath = osp.join(os.getcwd(), "openvino")


def load_class_map(file_path):
    """Returns class names map."""

    if file_path is not None and os.path.exists(file_path):
        with open(file_path, "r") as input_stream:
            data = json.load(input_stream)
            class_map = dict(enumerate(data))
    else:
        class_map = None

    return class_map


class recognition:
    def __init__(self):
        self.main()

    def main(self):
        classMapPath = osp.join(
            openvinoPath, "data", "dataset_classes", "msasl100.json"
        )
        self.class_map = load_class_map(classMapPath)
        if self.class_map is None:
            raise RuntimeError("Can't read {}".format(classMapPath))

        core = load_core()

        personDetectModel = "person-detection-asl-0001"
        person_detector = PersonDetector(
            osp.join(
                openvinoPath,
                "intel",
                personDetectModel,
                "FP16",
                f"{personDetectModel}.xml",
            ),
            "CPU",
            core,
            num_requests=2,
            output_shape=DETECTOR_OUTPUT_SHAPE,
        )

        actionRecognitionModel = "asl-recognition-0004"
        self.action_recognizer = ActionRecognizer(
            osp.join(
                openvinoPath,
                "intel",
                actionRecognitionModel,
                "FP16",
                f"{actionRecognitionModel}.xml",
            ),
            "CPU",
            core,
            num_requests=2,
            img_scale=ACTION_IMAGE_SCALE,
            num_classes=len(self.class_map),
        )

        self.person_tracker = Tracker(
            person_detector, TRACKER_SCORE_THRESHOLD, TRACKER_IOU_THRESHOLD
        )

        self.imageShape = [700, 580, 3]
        self.frameBuffer = []

    def getBatch(self, source):
        batchShape = [self.action_recognizer.input_length] + self.imageShape
        batchBufferSize = int(np.prod(batchShape))

        outBatch = Array("B", batchBufferSize, lock=True)
        frame = Array("B", int(np.prod(self.imageShape)), lock=True)

        with frame.get_lock():
            buffer = np.frombuffer(frame.get_obj(), dtype=np.uint8)
            np.copyto(buffer.reshape(self.imageShape), source)

        with frame.get_lock():
            inFrameBuffer = np.frombuffer(frame.get_obj(), dtype=np.uint8)
            frame = np.copy(inFrameBuffer.reshape(self.imageShape))

        self.frameBuffer.append(frame)
        if len(self.frameBuffer) > self.action_recognizer.input_length:
            self.frameBuffer = self.frameBuffer[-self.action_recognizer.input_length :]

        try:
            self.saveImageBuffer()
        except:
            pass

        if len(self.frameBuffer) == self.action_recognizer.input_length:
            with outBatch.get_lock():
                outBatchBuffer = np.frombuffer(outBatch.get_obj(), dtype=np.uint8)
                np.copyto(outBatchBuffer.reshape(batchShape), self.frameBuffer)
                return np.copy(outBatchBuffer.reshape(batchShape))
        else:
            return None

    def saveImageBuffer(self):
        grid = np.zeros((700 * 4, 580 * 4, 3), dtype=np.uint8)

        for i in range(4):
            for j in range(4):
                image = self.frameBuffer[i * 4 + j]
                grid[
                    i * 700 : (i + 1) * 700,
                    j * 580 : (j + 1) * 580,
                ] = image

        cv2.imwrite("test.jpg", grid)

    def process_frame(self, originalSource):
        source = cv2.cvtColor(originalSource, cv2.COLOR_RGB2BGR)

        active_object_id = -1
        tracker_labels_map = {}
        action_class_label = None
        scoreExists = False

        batch = self.getBatch(source)

        if batch is None:
            return None

        detections, tracker_labels_map = self.person_tracker.add_frame(
            source, len(OBJECT_IDS), tracker_labels_map
        )
        if detections is None:
            active_object_id = -1

        if len(detections) == 1:
            active_object_id = 0

        if active_object_id >= 0:
            cur_det = [det for det in detections if det.id == active_object_id]
            if len(cur_det) != 1:
                active_object_id = -1
                return None

            recognizer_result = self.action_recognizer(
                batch, cur_det[0].roi.reshape(-1)
            )

            if recognizer_result is not None:
                action_class_id = np.argmax(recognizer_result)
                action_class_label = (
                    self.class_map[action_class_id]
                    if self.class_map is not None
                    else action_class_id
                )

                action_class_score = np.max(recognizer_result)

                print(action_class_label, action_class_score)
                if action_class_score > 0.7:  # action_threshold
                    scoreExists = True

        person_pos = None

        if detections is not None:
            for det in detections:
                if det.id == active_object_id:
                    person_pos = det.roi[0]

                roi_color = (
                    (0, 255, 0) if active_object_id == det.id else (128, 128, 128)
                )
                border_width = 2 if active_object_id == det.id else 1
                person_roi = det.roi[0]
                cv2.rectangle(
                    source,
                    (person_roi[0], person_roi[1]),
                    (person_roi[2], person_roi[3]),
                    roi_color,
                    border_width,
                )
                cv2.putText(
                    source,
                    str(det.id),
                    (person_roi[0] + 10, person_roi[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    roi_color,
                    2,
                )

        if scoreExists:
            return {
                "frame": source,
                "label": action_class_label,
                "pos": [
                    str(person_pos[0]),
                    str(person_pos[1]),
                    str(person_pos[2]),
                    str(person_pos[3]),
                ],
            }
        else:
            return {
                "frame": source,
                "label": None,
                "pos": [
                    str(person_pos[0]),
                    str(person_pos[1]),
                    str(person_pos[2]),
                    str(person_pos[3]),
                ],
            }
