import cv2
import subprocess

# open videoX
device = 0
cap = cv2.VideoCapture(device) # NOTE: find correct video using v4l2-ctl --list-devices (get first videoX of usb cam list)

subprocess.run(["v4l2-ctl", "-d", "/dev/video" + str(device), "--set-ctrl=focus_automatic_continuous=1"])

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()