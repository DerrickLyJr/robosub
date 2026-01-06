import cv2
import glob
import numpy as np
from pathlib import Path
from computer_vision.main import image_capture
from computer_vision import config

cap0, cap1 = image_capture.camera_setup(config.LEFT_CAM_ID, config.RIGHT_CAM_ID)

# load calibration data
data = np.load(config.CALIBRATION_PATH + config.CALIBRATION_VAR_FILE)
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


while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    imageSize = frame0.shape[::-1]

    map1x, map1y = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, imageSize, cv2.CV_32FC1
    )

    map2x, map2y = cv2.initUndistortRectifyMap(
        K2, D2, R2, P2, imageSize, cv2.CV_32FC1
    )

    frame0 = cv2.remap(frame0,  map1x, map1y, cv2.INTER_LINEAR)
    frame1 = cv2.remap(frame1,  map1x, map1y, cv2.INTER_LINEAR)

    # grab values found with parameter_tuning.py
    channels = 1 # since using grayscale use 1
    num_disp = 128
    block_size = 10
    stereo = cv2.StereoSGBM_create(
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

    disparity = stereo.compute(frame0, frame1).astype(np.float32) / 16.0

    # median blur to remove noise
    disparity[disparity <= 0] = 0
    disparity = cv2.medianBlur(disparity.astype(np.float32), 5)

    map = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    cv2.line(map,(map.shape[1]//2,0), (map.shape[1]//2, map.shape[0]), 255, 1)
    cv2.line(map,(0,map.shape[0]//2), (map.shape[1], map.shape[0]//2), 255, 1)

    cv2.imshow(f"map: numDisp {num_disp}, blockSize {block_size}", map)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    X, Y, Z = points_3D[imageSize[0]//2, imageSize[1]//2]   # y first, then x

    size = 50//2
    center = map.shape[0]//2, map.shape[1]//2
    depth = points_3D[..., 2] # get only depth (Z)
    valid = np.isfinite(depth) & (depth > 0)
    valid_block = valid[center[0]-size:center[0]+size, center[1]-size:center[1]+size]
    depth_block = depth[center[0]-size:center[0]+size, center[1]-size:center[1]+size]
    center_depth = depth_block[valid_block]
    print("depth at center in meters: ", np.mean(center_depth)) ###################################### with current parameters give acc of +/- 20 cm. there is still a decent amount of block spots that affect readings

    # Get indices of valid depths
    ys, xs = np.where(valid)
    # Get depths of valid points
    valid_depths = depth[ys, xs]

    # Index of closest point among valid
    i = np.argmin(valid_depths)

    # Pixel coordinate of closest object
    y = ys[i]
    x = xs[i]

    # World coords
    X, Y, Z = points_3D[y, x]
    print(f"pixel coord x, y: {x}, {y}")
    print(f"x, y, z: {X}, {Y}, {Z} in meters")

cv2.destroyAllWindows()