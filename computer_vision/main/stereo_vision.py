import cv2
import glob
import numpy as np
from pathlib import Path
from computer_vision.main import image_capture
from computer_vision import config

cap0, cap1 = image_capture.camera_setup(config.LEFT_CAM_ID, config.RIGHT_CAM_ID)

if config.IS_FISHEYE:
    data = np.load(config.CALIBRATION_PATH + config.CALIBRATION_VAR_FILE_FISHEYE)
else:
    data = np.load(config.CALIBRATION_PATH + config.CALIBRATION_VAR_FILE)

# load calibration data
K1 = data["K1"]
D1 = data["D1"]
K2 = data["K2"]
D2 = data["D2"]
R = data["R"]
T = data["T"]
R1 = data["R1"]
R2 = data["R2"]
P1 = data["P1"]
P2 = data["P2"]
Q = data["Q"]


def get_disparity(display):
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    imageSize = frame0.shape[::-1]

    if config.IS_FISHEYE:
        map1x, map1y = cv2.fisheye.initUndistortRectifyMap(
            K1, D1, R1, P1[:, :3], imageSize, cv2.CV_32FC1
        )

        map2x, map2y = cv2.fisheye.initUndistortRectifyMap(
            K2, D2, R2, P2[:, :3], imageSize, cv2.CV_32FC1
        )
    else:
        map1x, map1y = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1, imageSize, cv2.CV_32FC1
        )

        map2x, map2y = cv2.initUndistortRectifyMap(
            K2, D2, R2, P2, imageSize, cv2.CV_32FC1
        )

    frame0 = cv2.remap(frame0,  map1x, map1y, cv2.INTER_LINEAR)
    frame1 = cv2.remap(frame1,  map2x, map2y, cv2.INTER_LINEAR)

    # grab values found with parameter_tuning.py
    channels = 1 # since using grayscale use 1
    num_disp = 128
    block_size = 10
    stereo = cv2.StereoSGBM_create(
    # left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * channels * block_size**2, # recommended openCV formula
        P2=32 * channels * block_size**2, # recommended openCV formula
        disp12MaxDiff=1,
        uniquenessRatio=2,
        speckleWindowSize=100,
        speckleRange=2
    )

    # disparity without wls seems to have better distance results but has more black holes in disparity map
    disparity = stereo.compute(frame0, frame1).astype(np.float32) / 16.0 # comment out this line and uncomment below to try wls filter

    # uncomment below and left_matcher above to use wls which highly reduces black blobs but seems to reduce distance accuarcy
    # right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # # compute raw disparities
    # dispL = left_matcher.compute(frame0, frame1).astype(np.int16)
    # dispR = right_matcher.compute(frame1, frame0).astype(np.int16)
    # #  WLS Filter
    # wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    # wls.setLambda(4000)          # smoothness strength
    # wls.setSigmaColor(1.5)       # edge sensitivity

    # disparity = wls.filter(dispL, frame0, disparity_map_right=dispR)

    # median blur to remove noise and black blobs
    disparity[disparity <= 0] = 0
    disparity = cv2.medianBlur(disparity.astype(np.float32), 5)

    # morphological closing to remove black blobs
    kernel = np.ones((3,3), np.uint8)
    disparity = cv2.morphologyEx(disparity, cv2.MORPH_CLOSE, kernel)

    map = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    if display:
        cv2.line(map,(map.shape[1]//2,0), (map.shape[1]//2, map.shape[0]), 255, 1)
        cv2.line(map,(0,map.shape[0]//2), (map.shape[1], map.shape[0]//2), 255, 1)

        cv2.imshow(f"map: numDisp {num_disp}, blockSize {block_size}", map)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        cv2.waitKey(1)

    return map, disparity, Q


def get_center_distance(display, window_square_size=50):
    map, disparity, Q = get_disparity(display)

    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    size = window_square_size//2
    center = map.shape[0]//2, map.shape[1]//2
    depth = points_3D[..., 2] # get only depth (Z)
    valid = np.isfinite(depth) & (depth > 0)
    valid_block = valid[center[0]-size:center[0]+size, center[1]-size:center[1]+size]
    depth_block = depth[center[0]-size:center[0]+size, center[1]-size:center[1]+size]
    center_depth = depth_block[valid_block]
    print("depth at center in meters: ", np.mean(center_depth))
    center_depth_val = np.mean(center_depth)
    return center_depth_val


def get_closest_obj_distance(display):
    map, disparity, Q = get_disparity(display)

    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    depth = points_3D[..., 2] # get only depth (Z)
    valid = np.isfinite(depth) & (depth > 0)

    # get indices of valid depths
    ys, xs = np.where(valid)
    # get depths of valid points
    valid_depths = depth[ys, xs]

    # index of closest point among valid
    i = np.argmin(valid_depths)

    # pixel coordinate of closest object
    y = ys[i]
    x = xs[i]

    # world coords
    X, Y, Z = points_3D[y, x]
    print(f"pixel coord x, y: {x}, {y}")
    print(f"x, y, z: {X}, {Y}, {Z} in meters")

    return X, Y, Z, x, y # meter coordinates X, Y, X, pixel coordinates x, y


if __name__ == "__main__":

    display = True

    while True:
        get_center_distance(display, 25)

    if display:
        cv2.destroyAllWindows()