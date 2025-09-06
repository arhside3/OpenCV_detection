import cv2
import numpy as np
import re3d
import pygame
import sys


CAM_RESOLUTION = (1280, 720)
ARUCO_SIZE = 0.0585
CAM_ID = 0

pygame.init()

graph_width, graph_height = 500, 400
graph_screen = pygame.display.set_mode((graph_width, graph_height))
pygame.display.set_caption("Углы наклона")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 80, 80)
GREEN = (80, 255, 80)
BLUE = (80, 120, 255)
YELLOW = (255, 200, 50)
PURPLE = (180, 80, 220)
CYAN = (50, 200, 200)
GRAY = (230, 230, 230)
LIGHT_GRAY = (240, 240, 240)
DARK_GRAY = (100, 100, 100)
BACKGROUND = (245, 247, 250)

GRADIENT_RED = [(255, 150, 150), (255, 100, 100), (220, 80, 80)]
GRADIENT_GREEN = [(150, 255, 150), (100, 255, 100), (80, 220, 80)]
GRADIENT_BLUE = [(150, 180, 255), (120, 160, 255), (100, 140, 220)]

rect_width = 300
rect_height = 30
rect_spacing = 50
start_x = 80
start_y = 80

current_ax, current_ay, current_az = 0, 0, 0
current_x, current_y, current_z = 0, 0, 0
target_ax, target_ay, target_az = 0, 0, 0
target_x, target_y, target_z = 0, 0, 0
animation_speed = 0.2 

pulse_value = 0
pulse_direction = 1

def smooth_interpolate(current, target, speed):
    return current + (target - current) * speed

def draw_gradient_rect(surface, color_list, rect, direction=0):
    """0 = горизонтальный, 1 = вертикальный"""
    if direction == 0:
        color_count = len(color_list)
        step_size = rect.width / (color_count - 1)
        for i in range(color_count - 1):
            color1 = color_list[i]
            color2 = color_list[i + 1]
            sub_rect = pygame.Rect(rect.x + i * step_size, rect.y, step_size, rect.height)
            pygame.draw.rect(surface, color1, sub_rect)
            pygame.draw.rect(surface, color2, 
                           (sub_rect.x + sub_rect.width // 2, sub_rect.y, 
                            sub_rect.width // 2, sub_rect.height))
    else:
        color_count = len(color_list)
        step_size = rect.height / (color_count - 1)
        for i in range(color_count - 1):
            color1 = color_list[i]
            color2 = color_list[i + 1]
            sub_rect = pygame.Rect(rect.x, rect.y + i * step_size, rect.width, step_size)
            pygame.draw.rect(surface, color1, sub_rect)
            pygame.draw.rect(surface, color2, 
                           (sub_rect.x, sub_rect.y + sub_rect.height // 2, 
                            sub_rect.width, sub_rect.height // 2))

def draw_digital_value(surface, value, x, y, color, label=""):
    font = pygame.font.SysFont('Courier New', 18, bold=True)
    if label:
        text = font.render(f"{label}: {value:7.2f}", True, color)
    else:
        text = font.render(f"{value:7.2f}", True, color)
    surface.blit(text, (x, y))
    return text.get_width()

def update_graphs():
    global current_ax, current_ay, current_az, current_x, current_y, current_z
    global pulse_value, pulse_direction
    
    # Обновляем пульсацию
    pulse_value += 0.05 * pulse_direction
    if pulse_value >= 1.0:
        pulse_value = 1.0
        pulse_direction = -1
    elif pulse_value <= 0.0:
        pulse_value = 0.0
        pulse_direction = 1
    
    current_ax = smooth_interpolate(current_ax, target_ax, animation_speed)
    current_ay = smooth_interpolate(current_ay, target_ay, animation_speed)
    current_az = smooth_interpolate(current_az, target_az, animation_speed)
    current_x = smooth_interpolate(current_x, target_x, animation_speed)
    current_y = smooth_interpolate(current_y, target_y, animation_speed)
    current_z = smooth_interpolate(current_z, target_z, animation_speed)
    
    for y in range(graph_height):
        color_factor = 1.0 - (y / graph_height) * 0.1
        color = (BACKGROUND[0] * color_factor, 
                BACKGROUND[1] * color_factor, 
                BACKGROUND[2] * color_factor)
        pygame.draw.line(graph_screen, color, (0, y), (graph_width, y))
    
    font_title = pygame.font.SysFont('Arial', 24, bold=True)
    title = font_title.render('Углы наклона', True, DARK_GRAY)
    title_shadow = font_title.render('Углы наклона', True, (200, 200, 220))
    graph_screen.blit(title_shadow, (graph_width//2 - title.get_width()//2 + 1, 16))
    graph_screen.blit(title, (graph_width//2 - title.get_width()//2, 15))
    
    values = [max(-1, min(1, val / 180)) for val in [current_ax, current_ay, current_az]]
    labels = ['X', 'Y', 'Z']
    colors = [RED, GREEN, BLUE]
    gradients = [GRADIENT_RED, GRADIENT_GREEN, GRADIENT_BLUE]
    
    font_section = pygame.font.SysFont('Arial', 20, bold=True)
    section_title = font_section.render('Углы наклона (°):', True, DARK_GRAY)
    graph_screen.blit(section_title, (20, start_y - 30))
    
    for i in range(3):
        y_pos = start_y + i * rect_spacing
        
        pygame.draw.rect(graph_screen, LIGHT_GRAY, 
                       (start_x - 5, y_pos - 5, rect_width + 10, rect_height + 10), 
                       border_radius=8)
        
        pygame.draw.rect(graph_screen, DARK_GRAY, 
                       (start_x, y_pos, rect_width, rect_height), 
                       2, border_radius=6)
        
        center_x = start_x + rect_width // 2
        pygame.draw.line(graph_screen, GRAY, (center_x, y_pos), (center_x, y_pos + rect_height), 2)
        
        fill_width = int(abs(values[i]) * (rect_width // 2))
        if values[i] > 0:
            fill_rect = pygame.Rect(center_x, y_pos, fill_width, rect_height)
            draw_gradient_rect(graph_screen, gradients[i], fill_rect, 0)
        elif values[i] < 0:
            fill_rect = pygame.Rect(center_x - fill_width, y_pos, fill_width, rect_height)
            draw_gradient_rect(graph_screen, gradients[i], fill_rect, 0)
        
        indicator_x = center_x + int(values[i] * (rect_width // 2))
        pygame.draw.circle(graph_screen, colors[i], (indicator_x, y_pos + rect_height // 2), 6)
        pygame.draw.circle(graph_screen, WHITE, (indicator_x, y_pos + rect_height // 2), 3)
        
        font = pygame.font.SysFont('Arial', 18, bold=True)
        value_text = font.render(f"{current_ax if i==0 else current_ay if i==1 else current_az:6.1f}°", True, DARK_GRAY)
        graph_screen.blit(value_text, (start_x + rect_width + 15, y_pos + 5))
        
        label_text = font.render(labels[i], True, DARK_GRAY)
        graph_screen.blit(label_text, (start_x - 30, y_pos + 5))
    
    coord_y = start_y + 3 * rect_spacing + 20

    
    coord_values = [current_ax, current_ay, current_az]
    coord_colors = [BLUE, BLUE, BLUE]
    coord_labels = ['X', 'Y', 'Z']
    
    for i in range(3):
        y_pos = coord_y + 25 + i * 25
        draw_digital_value(graph_screen, coord_values[i], 40, y_pos, coord_colors[i], coord_labels[i])
    
    font_status = pygame.font.SysFont('Arial', 16, bold=True)
    
    if marker_detected:
        pulse_color = (int(50 + 50 * pulse_value), int(200 + 55 * pulse_value), int(50 + 50 * pulse_value))
        status_text = font_status.render("Маркер обнаружен", True, pulse_color)
    else:
        pulse_color = (int(200 + 55 * pulse_value), int(50 + 50 * pulse_value), int(50 + 50 * pulse_value))
        status_text = font_status.render("Маркер не найден", True, pulse_color)
    
    pygame.draw.rect(graph_screen, LIGHT_GRAY, (10, graph_height - 40, status_text.get_width() + 20, 30), border_radius=5)
    graph_screen.blit(status_text, (20, graph_height - 35))
    
    pygame.display.flip()

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

marker_detected = False

update_graphs()


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
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
    
    marker_detected = False
    
    if all_ids:
        all_ids = np.concatenate(all_ids)
        marker_detected = True
        
        for i in range(len(all_ids)):
            try:
                idx = int(all_ids[i])
                cornersx = all_corners[i]
                
                position, mat = re3d.positionMarker(cornersx, ARUCO_SIZE, cameraMatrix)
                
                ax, ay, az = map(np.degrees, position[0])
                rx, ry, rz = map(np.degrees, position[1])

                
                target_ax, target_ay, target_az = ax, ay, az
                
                rvec, tvec = mat

                img = cv2.drawFrameAxes(img, cameraMatrix, np.array([]), rvec, tvec, 0.1, 5)

                img_pos = np.array(cornersx[0][0]).astype(np.int16)

                img = cv2.putText(
                    img,
                    f"Angle: {rz:.2f}",
                    [40, 30], cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 255), 2
                )
                
                marker_type = "6x6" if idx < 250 else "5x5"
                img = cv2.putText(
                    img,
                    f"ID:{idx} ({marker_type})",
                    [img_pos[0], img_pos[1] + 40], cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2
                )

            except Exception as e:
                print(f"Marker processing error: {e}")
                continue

    update_graphs()

    cv2.imshow("Display", img)

cap.release()
cv2.destroyAllWindows()
pygame.quit()
