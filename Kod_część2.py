import cv2
import os
import re

def create_video_from_images(image_folder='_cam2', output_file='cam2_29_06_2025_v2.mp4', fps=20):
    # Znajdź wszystkie pliki PNG w folderze
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    # Sortowanie po timestampie w nazwie pliku
    def extract_timestamp(name):
        match = re.search(r'camera_(\d+)\.png', name)
        return int(match.group(1)) if match else 0

    images.sort(key=extract_timestamp)

    if not images:
        print("Brak obrazów do przetworzenia.")
        return

    # Wczytaj pierwszy obraz, aby uzyskać rozmiar klatki
    first_frame_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_frame_path)
    height, width, layers = frame.shape
    size = (width, height)

    print(f"Rozpoczynanie zapisu filmu: {output_file} ({width}x{height} @ {fps} FPS)")
    
    # Inicjalizacja VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # dla plików MP4
    out = cv2.VideoWriter(output_file, fourcc, fps, size)

    for i, image_name in enumerate(images):
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Ostrzeżenie: Nie można załadować {image_name}")
            continue
        out.write(frame)
        print(f"[{i+1}/{len(images)}] Dodano: {image_name}")

    out.release()
    print("Zapis filmu zakończony.")

if __name__ == '__main__':
    create_video_from_images()
