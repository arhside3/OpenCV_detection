import cv2
import numpy as np
import re3d

CAM_RESOLUTION = (1280, 720)
ARUCO_SIZE = 0.0585
CAM_ID = 0

try:
    with open('calibration/param.txt') as f:
        cameraMatrix = eval(f.readline())
        distCoeffs = eval(f.readline())
    cameraMatrix = np.array(cameraMatrix, dtype=np.float64)
    distCoeffs = np.array(distCoeffs, dtype=np.float64)
except FileNotFoundError:
    print("Calibration file not found. Using default parameters.")
    cameraMatrix = np.array([
        [1000.0, 0.0, 960.0],
        [0.0, 1000.0, 540.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

w, h = CAM_RESOLUTION

cap = cv2.VideoCapture(CAM_ID)

if not cap.isOpened():
    print(f"Error: Could not open camera with ID={CAM_ID}")
    print("Trying other camera indexes...")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            CAM_ID = i
            print(f"Found camera with ID={i}")
            break

if not cap.isOpened():
    print("No camera found. Exiting.")
    exit(-1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv2.CAP_PROP_FPS, 30)

cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 250)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))

aruco_dict_6x6 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_dict_5x5 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

parameters = cv2.aruco.DetectorParameters()
detector_6x6 = cv2.aruco.ArucoDetector(aruco_dict_6x6, parameters)
detector_5x5 = cv2.aruco.ArucoDetector(aruco_dict_5x5, parameters)

print(f"Camera {CAM_ID} initialized successfully.")
print("Press 'q' to quit.")

while True:
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        continue

    try:
        img = cv2.fisheye.undistortImage(frame, cameraMatrix, D=distCoeffs, Knew=cameraMatrix)
    except Exception:
        img = frame.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    corners_6x6, ids_6x6, rejected_6x6 = detector_6x6.detectMarkers(gray)
    corners_5x5, ids_5x5, rejected_5x5 = detector_5x5.detectMarkers(gray)
    
    all_corners = []
    all_ids = []
    
    if ids_6x6 is not None:
        all_corners.extend(corners_6x6)
        all_ids.extend(ids_6x6)
    
    if ids_5x5 is not None:
        all_corners.extend(corners_5x5)
        all_ids.extend(ids_5x5)
    
    if all_ids:
        all_ids = np.concatenate(all_ids)
        
        for i in range(len(all_ids)):
            try:
                idx = int(all_ids[i])
                cornersx = all_corners[i]
                
                position, mat = re3d.positionMarker(cornersx, ARUCO_SIZE, cameraMatrix)
                
                x, y, z = position[0]
                rx, ry, rz = map(np.degrees, position[1])
                ax, ay, az = map(np.degrees, position[0])
                rvec, tvec = mat

                img = cv2.drawFrameAxes(img, cameraMatrix, np.array([]), rvec, tvec, 0.1, 5)

                img_pos = np.array(cornersx[0][0]).astype(np.int16)

                img = cv2.putText(
                    img,
                    f"x:{x:.2f}/y:{y:.2f}/z:{z:.2f}",
                    img_pos, cv2.FONT_HERSHEY_SIMPLEX, 0, (255, 0, 255), 2
                )

                img = cv2.putText(
                    img,
                    f"Angle: {rz:.2f}",
                    [40, 30], cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 255), 2
                )
                
                img = cv2.putText(
                    img,
                    f"x:{ax:.2f}/y:{ay:.2f}/z:{az:.2f}",
                    [40, 80], cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 255), 2
                )
                

                # Also display marker type and ID
                marker_type = "6x6" if idx < 250 else "5x5"
                img = cv2.putText(
                    img,
                    f"ID:{idx} ({marker_type})",
                    [img_pos[0], img_pos[1] + 40], cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 255), 2
                )

            except Exception as e:
                print(f"Marker processing error: {e}")
                continue

    cv2.imshow("Display", img)

cap.release()
cv2.destroyAllWindows()