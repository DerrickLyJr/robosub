import cv2
import subprocess
from computer_vision import config

# open videoX
device0 = config.LEFT_CAM_ID
device1 = config.RIGHT_CAM_ID
cap0 = cv2.VideoCapture(device0) # NOTE: find correct video using v4l2-ctl --list-devices (get first videoX of usb cam list)
cap1 = cv2.VideoCapture(device1)

# enable auto focus (to allow for capture control)
subprocess.run(["v4l2-ctl", "-d", "/dev/video" + str(device0), "--set-ctrl=focus_automatic_continuous=1"]) # 1 = Enable, 0 = disable
subprocess.run(["v4l2-ctl", "-d", "/dev/video" + str(device1), "--set-ctrl=focus_automatic_continuous=1"])

# disable auto exposure
subprocess.run(["v4l2-ctl", "-d", "/dev/video" + str(device0), "--set-ctrl=auto_exposure=1"]) # 1 = Manual Mode, 3 = Aperture Priority Mode
subprocess.run(["v4l2-ctl", "-d", "/dev/video" + str(device1), "--set-ctrl=auto_exposure=1"])

# disable auto white balance
subprocess.run(["v4l2-ctl", "-d", "/dev/video" + str(device0), "--set-ctrl=white_balance_automatic=0"]) # 1 = on, 0 = off
subprocess.run(["v4l2-ctl", "-d", "/dev/video" + str(device1), "--set-ctrl=white_balance_automatic=0"])

while(True):
    # Capture frame-by-frame
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    # Display the resulting frame
    cv2.imshow('frame0', frame0)
    cv2.imshow('frame1', frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap0.release()
cap1.release()
cv2.destroyAllWindows()