from computer_vision.main import image_capture
from computer_vision import config
import cv2
import os

cap0, cap1 = image_capture.camera_setup(config.LEFT_CAM_ID, config.RIGHT_CAM_ID)

img_count = 0

while(True):
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    # Display the resulting frame
    cv2.imshow('frame0', frame0)
    cv2.imshow('frame1', frame1)

    k = cv2.waitKey(1) & 0xFF

    if k == ord('c'):
        l_path = config.CALIBRATION_TEST_IN_LEFT_PATH + f"left_test_img{img_count}.png"
        r_path = config.CALIBRATION_TEST_IN_RIGHT_PATH + f"right_test_img{img_count}.png"
        cv2.imwrite(l_path, frame0)
        cv2.imwrite(r_path, frame1)

        print(f"Image saved as {os.path.abspath(l_path)}")
        print(f"Image saved as {os.path.abspath(r_path)}")

        img_count+=1

    if k == ord('q'):
        break

# When everything done, release the capture
cap0.release()
cap1.release()
cv2.destroyAllWindows()