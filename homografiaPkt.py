import cv2
import numpy as np

# Globalna lista do przechowywania wszystkich punktów
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    """Funkcja zwrotna obsługująca zdarzenia myszy"""
    if event == cv2.EVENT_LBUTTONDOWN:
        # Wypisz współrzędne w konsoli po kliknięciu lewym przyciskiem
        print(f"Clicked point: ({x}, {y})")
        clicked_points.append((x, y))
        
        # Aktualizuj obrazek, aby pokazać kliknięty punkt
        cv2.circle(display_img, (x, y), 4, (0, 0, 255), -1)  # Czerwone kółko
        cv2.putText(display_img, str(len(clicked_points)-1), (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 2)  # Biały numer
        
        # Rysuj linie między punktami, jeśli jest co najmniej 2 punktów
        if len(clicked_points) >= 2:
            points = np.array(clicked_points, dtype=np.int32)
            cv2.polylines(display_img, [points], isClosed=(len(clicked_points) >= 3), 
                         color=(0, 255, 0), thickness=2)  # Zielona linia
        
        # Pokaż zaktualizowany obraz
        cv2.imshow("Click to capture coordinates", display_img)

# Ścieżka do pliku wideo
video_path = "/mnt/c/Users/drvik/OneDrive/Pulpit/Yolo/V6/output_video1.mp4"

# Otwórz plik wideo
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Cannot open video: {video_path}")
    exit()

# Wczytaj pierwszą klatkę
ret, frame = cap.read()
if not ret:
    print("Failed to read frame from video")
    cap.release()
    exit()

# Stwórz kopię pierwszej klatki
display_img = frame.copy()

cv2.namedWindow("Click to capture coordinates")
cv2.setMouseCallback("Click to capture coordinates", mouse_callback)
cv2.imshow("Click to capture coordinates", display_img)

print("Click on the image to capture coordinates. Press 'q' to quit, 'c' to clear all points.")
print("All clicked points will be printed to the console.")

while True:
    key = cv2.waitKey(1) & 0xFF
    
    # Wciśnij 'q' aby wyjść
    if key == ord('q'):
        break
        
    # Wciśnij 'c' aby wyczyścić punkty
    elif key == ord('c'):
        clicked_points.clear()
        display_img = frame.copy()
        cv2.imshow("Click to capture coordinates", display_img)
        print("Points cleared")
        
    # Wciśnij 's' aby zapisać punkty
    elif key == ord('s'):
        print("\nFinal points list:")
        print(clicked_points)
        print(f"\nConfig format: 'img_points': {clicked_points},")
        
cap.release()
cv2.destroyAllWindows()

# Wypisz punkty
if clicked_points:
    print("\nFinal points list:")
    print(clicked_points)
    print(f"\nConfig format: 'img_points': {clicked_points},")