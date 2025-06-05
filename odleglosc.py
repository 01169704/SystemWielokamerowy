import cv2
import numpy as np
import math

clicked_points = []

def calculate_distances(p1, p2):
    """Oblicza odległości w osiach X i Y oraz odległość euklidesową między dwoma punktami"""
    dx = p2[0] - p1[0]  # różnica w osi X
    dy = p2[1] - p1[1]  # różnica w osi Y
    euclidean = math.sqrt(dx**2 + dy**2)  # odległość euklidesowa
    return dx, dy, euclidean

def mouse_callback(event, x, y, flags, param):
    """Funkcja zwrotna obsługująca zdarzenia myszy"""
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked point: ({x}, {y})")
        clicked_points.append((x, y))
        
        if len(clicked_points) == 1:
            cv2.circle(display_img, (x, y), 12, (255, 0, 0), -1)
            cv2.putText(display_img, "BASE", (x + 15, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 2)
            print("Base point set!")
        else:
            base_point = clicked_points[0]
            dx, dy, euclidean = calculate_distances(base_point, (x, y))
            
            cv2.circle(display_img, (x, y), 10, (0, 0, 255), -1)
            
            point_num = len(clicked_points) - 1
            """
            cv2.putText(display_img, f"{point_num}", (x, y + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(display_img, f"X:{dx:+.0f}", (x + 15, y - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(display_img, f"Y:{dy:+.0f}", (x + 15, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(display_img, f"D:{euclidean:.0f}", (x + 15, y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            cv2.line(display_img, base_point, (x, y), (0, 255, 0), 2)
            """
            print(f"Point {point_num}: X={dx:+.0f}px, Y={dy:+.0f}px, Distance={euclidean:.1f}px")
        
        cv2.imshow("Distance Measurement Tool", display_img)

# Ścieżka do pliku obrazu
image_path = "/mnt/c/Users/drvik/OneDrive/Pulpit/Yolo/V2/gora.png"

# Wczytaj obraz
frame = cv2.imread(image_path)
if frame is None:
    print(f"Cannot open image: {image_path}")
    exit()

display_img = frame.copy()

cv2.namedWindow("Distance Measurement Tool")
cv2.setMouseCallback("Distance Measurement Tool", mouse_callback)
cv2.imshow("Distance Measurement Tool", display_img)

print("=== DISTANCE MEASUREMENT TOOL ===")
print("1. Click to set BASE point (first click)")
print("2. Click additional points to measure X, Y distances from base")
print("Press 'q' to quit, 'c' to clear all points, 's' to save points")
print("X: horizontal distance (+right, -left)")
print("Y: vertical distance (+down, -up)")
print("D: total euclidean distance")
print("=====================================")

while True:
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
        
    elif key == ord('c'):
        clicked_points.clear()
        display_img = frame.copy()
        cv2.imshow("Distance Measurement Tool", display_img)
        print("All points cleared - click to set new base point")
        
    elif key == ord('s'):
        if clicked_points:
            print("\n=== MEASUREMENT RESULTS ===")
            print(f"Base point: {clicked_points[0]}")
            if len(clicked_points) > 1:
                base_point = clicked_points[0]
                for i, point in enumerate(clicked_points[1:], 1):
                    dx, dy, euclidean = calculate_distances(base_point, point)
                    print(f"Point {i}: {point} - X: {dx:+.0f}px, Y: {dy:+.0f}px, Distance: {euclidean:.1f}px")
            print(f"\nAll points: {clicked_points}")
            print(f"Config format: 'img_points': {clicked_points},")
            print("==========================")
        else:
            print("No points to save!")

cv2.destroyAllWindows()

if clicked_points:
    print("\n=== FINAL RESULTS ===")
    print(f"Base point: {clicked_points[0]}")
    if len(clicked_points) > 1:
        base_point = clicked_points[0]
        for i, point in enumerate(clicked_points[1:], 1):
            dx, dy, euclidean = calculate_distances(base_point, point)
            print(f"Point {i}: {point} - X: {dx:+.0f}px, Y: {dy:+.0f}px, Distance: {euclidean:.1f}px")
    print(f"\nAll points: {clicked_points}")
    print(f"Config format: 'img_points': {clicked_points},")
    print("====================")