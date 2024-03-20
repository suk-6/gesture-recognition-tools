import os
import os.path as osp

from gesture_recognition_tools.recognition import recognition


class main:
    def __init__(self):
        self.openvinoPath = osp.join(os.getcwd(), "openvino")
        self.downloadFromOpenVINO()

        self._model = recognition()

    def downloadFromOpenVINO(self):
        """Downloads models from OpenVINO model zoo."""
        if not osp.exists(osp.join(self.openvinoPath, "intel", "asl-recognition-0004")):
            os.system(
                f"omz_downloader --name asl-recognition-0004 -o {self.openvinoPath}"
            )

        if not osp.exists(
            osp.join(self.openvinoPath, "intel", "person-detection-asl-0001")
        ):
            os.system(
                f"omz_downloader --name person-detection-asl-0001 -o {self.openvinoPath}"
            )

        if not osp.exists(osp.join(self.openvinoPath, "data")):
            os.system(f"omz_data_downloader -o {self.openvinoPath}")

    @property
    def model(self):
        return self._model
