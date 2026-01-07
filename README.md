# robosub
The software for Competitive Robotics Underwater Autonomous Vehicle



# Computer Vision Documentation

## Camera Setup

### Installing Utilities
Reference the document U20CAM-9281M-V1.1 for details of full camera operation and setup. 
1. To operate camera with GUI on Linux, install Guvcview with ```sudo apt install guvcview```. 
2. For command line camera control, install V4L with ```sudo apt-get install v4l-utils```.

### Test Trigger Mode
1. To enable external trigger with Guvcview, run using ```sudo guvcview```. 
2. Next, enable "Focus, Automatic Continous" setting. If no microcontroller is currently triggering FSIN then camera view will freeze and may prompt a force shutdown, if so click on "Wait". This is expected since the camera is not getting prompted to get a new frame. 
3. Now attach microcontroller with triggering script running. The cameras should now be capturing frames when triggered. Note, the FPS of camera may or may not match triggering rate. Testing shows that guvcview did not show camera FPS greater than 16 FPS. 


### Finding camera Id's
Next find correct camera device id's. Make sure both cameras are connected and microcontroller is triggering FSIN.

1. Open Guvcview and go to Video Controls tab near the top.
2. Click on Device dropdown. Treat the list of devices with index values starting with the top as index 0.
3. Click through the devices to find the correct camera device(s). If a popup says open new window appears click "New". Correct video devices will likely be every 2 devices such as index 0 and 2 or 1 and 3.
4. Once correct cameras were found, in a terminal run ```v4l2-ctl --list-devices```. The /dev/videoX will correspond to the index you found earlier. For 2 cameras the /dev/videoX and /dev/videoY will be the corresponding X and Y index you found earlier. 
5. Open config.py and change LEFT_CAM_ID and RIGHT_CAM_ID to correct id value. Note left is the camera that is left when lens face opposite of your view and usb connector pins face you. 
6. From robosub (project root) run ```python -m computer_vision.test.test_two_camera_stream```. Verify that the "Left" and "Right" windows are marked correctly. 

## Camera Calibration

### Capture Calibration Images

1. Obtain checkered board. Previously tested on 10 x 7. Go to computer_vision/calibration/capture_calibration_images.py and update CHECKER_BOARD_ROWS and CHECKER_BOARD_COLUMNS. Note only count inner rows and columns for example a 10x7 board will actually be rows = 6, comuns = 9. Also update CHECKER_BOARD_SQUARE_SIZE in meters.
2. From project root run ```python -m computer_vision.calibration.capture_calibration_images```.
3. You will see a live feed of the camera. Hold the checkerboard up to the cameras. Make sure the board covers 30-70% of screen and all squares are visible. Press "c" on keyboard to capture image. 
4. If inner corners of checkerboard are found press "y" to save image for calibration, if it doesn't press "n" to discard. 
5. Continue to capture test images until script ends. Capture images with boards in different positions and orientations. Images will be saved to calibration_images_left/ and calibration_images_right/


### Camera and Stereo Calibration

1. Run ```python -m computer_vision.calibration.opencv_fisheye_calibration```. This will produce stereo_calibration_fisheye.npz which contains calibration variables. 
3. Find a good scene to use for disparity map testing. The scene should have diverse textures and good lighting. To capture scene for testing and tuning run ```python -m computer_vision.calibration.capture_tuning_images```.
4. Press "c" to capture scene. You can capture multiple scenes to use for testing with this script. Press "q" to quit. Scenes are saved to calibration_tuning_images/in/left and calibration_tuning_images/in/right/. 
5. 