import cv2
import glob
import numpy as np
from computer_vision import config

# tunning strategy
"""
1) Determine a good default state as the base for incremental changes
2) One parameter at a time increment values, select the best choice, and use
    best choice as new current/default. Repeat this step for each parameter
"""

# default
current_minDisparity=0
current_numDisparities=64
current_blockSize=15
current_disp12MaxDiff=1
current_uniquenessRatio=7
current_speckleWindowSize=100
current_speckleRange=2

current_map = None

test_image_index = 0

title = (
    f"numDisp {current_numDisparities}, " + \
    f"blockSize {current_blockSize}, " + \
    f"MaxDiff {current_disp12MaxDiff}, " + \
    f"uniqRatio {current_uniquenessRatio}, " + \
    f"specWindSize {current_speckleWindowSize}, " + \
    f"specRange {current_speckleRange}"
)


def rectify_images():
    images_names = sorted(glob.glob(config.CALIBRATION_IMAGES_LEFT_PATH + "*"))
    im = cv2.imread(images_names[0], cv2.IMREAD_GRAYSCALE)
    imageSize = im.shape[::-1]

    # load the calibration results
    if config.IS_FISHEYE:
        data = np.load(config.CALIBRATION_PATH + config.CALIBRATION_VAR_FILE_FISHEYE)
    else:
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

    # do remapping based on grid findings
    imgs_l = sorted(glob.glob(config.CALIBRATION_TEST_IN_LEFT_PATH + "*"))
    imgs_r = sorted(glob.glob(config.CALIBRATION_TEST_IN_RIGHT_PATH + "*"))

    i = 0
    for img_l, img_r in zip(imgs_l, imgs_r):
        imL = cv2.imread(img_l, cv2.IMREAD_GRAYSCALE)
        imR = cv2.imread(img_r, cv2.IMREAD_GRAYSCALE)

        left_rect  = cv2.remap(imL,  map1x, map1y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(imR, map2x, map2y, cv2.INTER_LINEAR)

        cv2.imwrite(config.CALIBRATION_TEST_OUT_LEFT_PATH + f"left_rectified_img{i}.png", left_rect)
        cv2.imwrite(config.CALIBRATION_TEST_OUT_RIGHT_PATH + f"right_rectified_img{i}.png", right_rect)
        i+=1

    print("Baseline (m):", np.linalg.norm(T))


def build_disparity_map(
        minDisparity,
        numDisparities,
        blockSize,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=120,
        speckleRange=2,
        rect_img_index=0
    ):

    # create depth map
    img_path0 = config.CALIBRATION_TEST_OUT_LEFT_PATH + f"left_rectified_img{rect_img_index}.png"
    img_path1 = config.CALIBRATION_TEST_OUT_RIGHT_PATH + f"right_rectified_img{rect_img_index}.png"

    img0 = cv2.imread(img_path0, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)

    # stereo = cv2.StereoSGBM_create(numDisparities=128, blockSize=15)
    channels = 1 # since using grayscale use 1
    stereo = cv2.StereoSGBM_create(
        minDisparity=minDisparity,
        numDisparities=numDisparities,
        blockSize=blockSize,
        P1=8 * channels * blockSize**2, # recommended openCV formula
        P2=32 * channels * blockSize**2, # recommended openCV formula
        disp12MaxDiff=disp12MaxDiff,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange
    )
    disparity = stereo.compute(img0, img1).astype(np.float32) / 16.0
    map = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return map


def tune_numDisparities(rect_img_index=0):
    # how far the matcher searches for a match
    # 16 - 512, val must be div by 16
    i = 16
    while i <= 512:
        current_numDisparities = i

        current_map = build_disparity_map(
            minDisparity=current_minDisparity,
            numDisparities=current_numDisparities,
            blockSize=current_blockSize,
            disp12MaxDiff=current_disp12MaxDiff,
            uniquenessRatio=current_uniquenessRatio,
            speckleWindowSize=current_speckleWindowSize,
            speckleRange=current_speckleRange, 
            rect_img_index=rect_img_index
        )

        title = (
        f"numDisp {current_numDisparities}, " + \
        f"blockSize {current_blockSize}, " + \
        f"MaxDiff {current_disp12MaxDiff}, " + \
        f"uniqRatio {current_uniquenessRatio}, " + \
        f"specWindSize {current_speckleWindowSize}, " + \
        f"specRange {current_speckleRange}"
        )
        cv2.imshow(title, current_map)
        i += i
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    current_numDisparities = int(input("best numDisp (int): "))



def tune_blockSize(rect_img_index=0):
    # window size for block matching
    # 1 - 15
    for i in range(1, 16, 2):
        current_blockSize = i

        current_map = build_disparity_map(
            minDisparity=current_minDisparity,
            numDisparities=current_numDisparities,
            blockSize=current_blockSize,
            disp12MaxDiff=current_disp12MaxDiff,
            uniquenessRatio=current_uniquenessRatio,
            speckleWindowSize=current_speckleWindowSize,
            speckleRange=current_speckleRange,
            rect_img_index=rect_img_index
        )

        title = (
        f"numDisp {current_numDisparities}, " + \
        f"blockSize {current_blockSize}, " + \
        f"MaxDiff {current_disp12MaxDiff}, " + \
        f"uniqRatio {current_uniquenessRatio}, " + \
        f"specWindSize {current_speckleWindowSize}, " + \
        f"specRange {current_speckleRange}"
        )
        cv2.imshow(title, current_map)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    current_blockSize = int(input("best blockSize (int): "))



def tune_MaxDiff(rect_img_index=0):
    # sets the maximum allowed difference between left and right image window
    # 0 - 10
    for i in range(1, 11):
        current_disp12MaxDiff = i

        current_map = build_disparity_map(
            minDisparity=current_minDisparity,
            numDisparities=current_numDisparities,
            blockSize=current_blockSize,
            disp12MaxDiff=current_disp12MaxDiff,
            uniquenessRatio=current_uniquenessRatio,
            speckleWindowSize=current_speckleWindowSize,
            speckleRange=current_speckleRange,
            rect_img_index=rect_img_index
        )

        title = (
        f"numDisp {current_numDisparities}, " + \
        f"blockSize {current_blockSize}, " + \
        f"MaxDiff {current_disp12MaxDiff}, " + \
        f"uniqRatio {current_uniquenessRatio}, " + \
        f"specWindSize {current_speckleWindowSize}, " + \
        f"specRange {current_speckleRange}"
        )
        cv2.imshow(title, current_map)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    current_disp12MaxDiff = int(input("best MaxDiff (int): "))


def tune_uniquenessRatio(rect_img_index=0):
    # rejects ambiguous matches
    # 1 - 15
    for i in range(1, 16, 2):
        current_uniquenessRatio = i

        current_map = build_disparity_map(
            minDisparity=current_minDisparity,
            numDisparities=current_numDisparities,
            blockSize=current_blockSize,
            disp12MaxDiff=current_disp12MaxDiff,
            uniquenessRatio=current_uniquenessRatio,
            speckleWindowSize=current_speckleWindowSize,
            speckleRange=current_speckleRange,
            rect_img_index=rect_img_index
        )

        title = (
        f"numDisp {current_numDisparities}, " + \
        f"blockSize {current_blockSize}, " + \
        f"MaxDiff {current_disp12MaxDiff}, " + \
        f"uniqRatio {current_uniquenessRatio}, " + \
        f"specWindSize {current_speckleWindowSize}, " + \
        f"specRange {current_speckleRange}"
        )
        cv2.imshow(title, current_map)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    current_uniquenessRatio = int(input("best uniqRatio (int): "))


def tune_speckleWindowSize(rect_img_index=0):
    # minimum connected region size to keep
    # 50 - 200
    for i in range(50, 200, 25):
        current_speckleWindowSize = i

        current_map = build_disparity_map(
            minDisparity=current_minDisparity,
            numDisparities=current_numDisparities,
            blockSize=current_blockSize,
            disp12MaxDiff=current_disp12MaxDiff,
            uniquenessRatio=current_uniquenessRatio,
            speckleWindowSize=current_speckleWindowSize,
            speckleRange=current_speckleRange,
            rect_img_index=rect_img_index
        )

        title = (
        f"numDisp {current_numDisparities}, " + \
        f"blockSize {current_blockSize}, " + \
        f"MaxDiff {current_disp12MaxDiff}, " + \
        f"uniqRatio {current_uniquenessRatio}, " + \
        f"specWindSize {current_speckleWindowSize}, " + \
        f"specRange {current_speckleRange}"
        )
        cv2.imshow(title, current_map)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    current_speckleWindowSize = int(input("best specWindSize (int): "))


def tune_speckleRange(rect_img_index=0):
    # minimum connected region size to keep
    # 1 - 5
    for i in range(1, 5):
        current_speckleRange = i

        current_map = build_disparity_map(
            minDisparity=current_minDisparity,
            numDisparities=current_numDisparities,
            blockSize=current_blockSize,
            disp12MaxDiff=current_disp12MaxDiff,
            uniquenessRatio=current_uniquenessRatio,
            speckleWindowSize=current_speckleWindowSize,
            speckleRange=current_speckleRange,
            rect_img_index=rect_img_index
        )

        title = (
        f"numDisp {current_numDisparities}, " + \
        f"blockSize {current_blockSize}, " + \
        f"MaxDiff {current_disp12MaxDiff}, " + \
        f"uniqRatio {current_uniquenessRatio}, " + \
        f"specWindSize {current_speckleWindowSize}, " + \
        f"specRange {current_speckleRange}"
        )
        cv2.imshow(title, current_map)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    current_speckleRange = int(input("best specRange (int): "))


if __name__ == "__main__":

    test_image_index = 0 # change to test different test image

    # must run rectify_images before any build_disparity_map()
    rectify_images()

    current_map = build_disparity_map(
        minDisparity=current_minDisparity,
        numDisparities=current_numDisparities,
        blockSize=current_blockSize,
        disp12MaxDiff=current_disp12MaxDiff,
        uniquenessRatio=current_uniquenessRatio,
        speckleWindowSize=current_speckleWindowSize,
        speckleRange=current_speckleRange
    )
    
    # 1
    cv2.imshow("default " + title, current_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 2
    tune_numDisparities(test_image_index)
    tune_blockSize(test_image_index)
    tune_MaxDiff(test_image_index)
    tune_uniquenessRatio(test_image_index)
    tune_speckleWindowSize(test_image_index)
    tune_speckleRange(test_image_index)

    cv2.imshow("result " + title, current_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()