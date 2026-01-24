import cv2
import glob
import numpy as np
from computer_vision import config

def calibrate_camera(images_folder, calibration_parameters):
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv2.imread(imname, cv2.IMREAD_GRAYSCALE)
        images.append(im)
 
    # criteria used by checkerboard pattern detector.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    rows = calibration_parameters[0] # number of checkerboard rows.
    columns = calibration_parameters[1] # number of checkerboard columns.
    world_scaling = calibration_parameters[2] # size of one chessboard square in meters
    shape = (rows, columns)
 
    # coordinates of squares in the checkerboard world space
    objp = np.zeros((1, rows * columns, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp
 
    # frame dimensions
    width = images[0].shape[1]
    height = images[0].shape[0]
 
    # pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.
 
    # coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    for frame in images:
 
        # find the checkerboard
        ret, corners = cv2.findChessboardCorners(frame, shape, None)
 
        if ret == True:
 
            # convolution size used to improve corner detection. Don't make this too large.
            conv_size = (3, 3)
 
            # opencv can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(frame, corners, conv_size, (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, shape, corners, ret)
 
            objpoints.append(objp)
            imgpoints.append(corners)

    # Ensure correct fisheye shapes
    objpoints = [op.reshape(1, -1, 3) for op in objpoints]
    imgpoints = [ip.reshape(1, -1, 2) for ip in imgpoints]

    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
            cv2.fisheye.CALIB_FIX_SKEW + \
            cv2.fisheye.CALIB_CHECK_COND

    ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints, imgpoints, (width, height), None, None, None, None,
    flags, criteria
)
 
    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coeffs:', dist)
    print('Rs:\n', rvecs)
    print('Ts:\n', tvecs)
 
    return mtx, dist


def stereo_calibrate(mtx1, dist1, mtx2, dist2, left_images, right_images, calibration_parameters):
    # read the synched frames
    images_names_l = glob.glob(left_images)
    images_names_l = sorted(images_names_l)
    c1_images_names = images_names_l
    
    images_names_r = glob.glob(right_images)
    images_names_r = sorted(images_names_r)
    c2_images_names = images_names_r
 
    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv2.imread(im1, cv2.IMREAD_GRAYSCALE)
        c1_images.append(_im)
 
        _im = cv2.imread(im2, cv2.IMREAD_GRAYSCALE)
        c2_images.append(_im)
 
    # change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
    rows = calibration_parameters[0] # number of checkerboard rows.
    columns = calibration_parameters[1] # number of checkerboard columns.
    world_scaling = calibration_parameters[2] # size of one chessboard square in meters
    shape = (rows, columns)
 
    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    # frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    # pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    # coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    for frame1, frame2 in zip(c1_images, c2_images):
        c_ret1, corners1 = cv2.findChessboardCorners(frame1, shape, None)
        c_ret2, corners2 = cv2.findChessboardCorners(frame2, shape, None)
 
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv2.cornerSubPix(frame1, corners1, (3, 3), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(frame2, corners2, (3, 3), (-1, -1), criteria)
 
            cv2.drawChessboardCorners(frame1, shape, corners1, c_ret1)
            cv2.drawChessboardCorners(frame2, shape, corners2, c_ret2)
 
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    # Ensure correct fisheye shapes
    objpoints = [op.reshape(1, -1, 3) for op in objpoints]
    imgpoints_left = [ip.reshape(1, -1, 2) for ip in imgpoints_left]
    imgpoints_right = [ip.reshape(1, -1, 2) for ip in imgpoints_right]


    result = cv2.fisheye.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        mtx1, dist1,
        mtx2, dist2,
        (width, height),
        criteria=criteria,
        flags=cv2.fisheye.CALIB_FIX_INTRINSIC
    )

    rms = result[0]
    K1 = result[1]
    D1 = result[2]
    K2 = result[3]
    D2 = result[4]
    R  = result[5]
    T  = result[6]

    print("K1:\n", K1)
    print("D1:\n", D1)
    print("K2:\n", K2)
    print("D2:\n", D2)
    print("The overall reprojection error (RMS)", rms)

    return rms, K1, D1, K2, D2, R, T


if __name__ == "__main__":

    rows = config.CHECKER_BOARD_ROWS # number of checkerboard rows.
    columns = config.CHECKER_BOARD_COLUMNS # number of checkerboard columns.
    world_scaling = config.CHECKER_BOARD_SQUARE_SIZE # size of one chessboard square in meters
    calibration_parameters = (rows, columns, world_scaling)

    images_names = sorted(glob.glob(config.CALIBRATION_IMAGES_LEFT_PATH + "*"))
    im = cv2.imread(images_names[0], cv2.IMREAD_GRAYSCALE)
    imageSize = im.shape[::-1]
    
    mtx1, dist1 = calibrate_camera(config.CALIBRATION_IMAGES_LEFT_PATH + '*', calibration_parameters)
    mtx2, dist2 = calibrate_camera(config.CALIBRATION_IMAGES_RIGHT_PATH + '*', calibration_parameters)

    ret, K1, D1, K2, D2, R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, config.CALIBRATION_IMAGES_LEFT_PATH + '*', config.CALIBRATION_IMAGES_RIGHT_PATH + '*', calibration_parameters)

    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K1, D1, K2, D2, imageSize, R, T, 
        cv2.CALIB_ZERO_DISPARITY,
        imageSize, 
        balance=0.0, 
        fov_scale=1.0
    )

    # save calibration data for later use in stereo vision
    np.savez(
        config.CALIBRATION_PATH + config.CALIBRATION_VAR_FILE_FISHEYE,
        K1=K1,
        D1=D1,
        K2=K2,
        D2=D2,
        R=R,
        T=T,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q
    )