from pathlib import Path
import os

# global computer_vision variables

# paths end with /
# paths are absolute paths since they are based off of PROJECT_ROOT_PATH
PROJECT_ROOT_PATH = str(Path(__file__).resolve().parent) + "/"

CALIBRATION_PATH = os.path.join(PROJECT_ROOT_PATH, "calibration/")

CALIBRATION_IMAGES_LEFT_PATH = os.path.join(CALIBRATION_PATH, "calibration_images_left/")
CALIBRATION_IMAGES_RIGHT_PATH = os.path.join(CALIBRATION_PATH, "calibration_images_right/")

CALIBRATION_TEST_IN_LEFT_PATH = os.path.join(CALIBRATION_PATH, "calibration_tuning_images/in/left/")
CALIBRATION_TEST_IN_RIGHT_PATH = os.path.join(CALIBRATION_PATH, "calibration_tuning_images/in/right/")
CALIBRATION_TEST_OUT_LEFT_PATH = os.path.join(CALIBRATION_PATH, "calibration_tuning_images/out/left/")
CALIBRATION_TEST_OUT_RIGHT_PATH = os.path.join(CALIBRATION_PATH, "calibration_tuning_images/out/right/")

CALIBRATION_VAR_FILE = "stereo_calibration_pinhole.npz"
CALIBRATION_VAR_FILE_FISHEYE = "stereo_calibration_fisheye.npz"

LEFT_CAM_ID = 0
RIGHT_CAM_ID = 2
IS_FISHEYE = True