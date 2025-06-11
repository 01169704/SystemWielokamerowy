import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from scipy.spatial.distance import cdist
from collections import defaultdict, deque

# Wspólne parametry globalne
s = 0.3  # Współczynnik skalujący
data_dir = "/mnt/c/Users/drvik/OneDrive/Pulpit/Yolo/V6"
model_path = "yolov12x.pt"
conf_threshold = 0.3
homografiaX = 1200  # Szerokość płótna dla połączonej homografii
homografiaY = 900  # Wysokość płótna dla połączonej homografii
offsetX = 1300
offsetY = 2000

# 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
TARGET_CLASS_IDS = [0, 2, 3, 5, 7]
CLASS_NAMES = {0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

SIMILAR_CLASSES = {
    frozenset([0, 3]): "person_motorcycle"
}

CLASS_COMPATIBILITY = {
    2: [2, 5, 7],  # car może być pomylony z bus, truck
    5: [2, 5, 7],  # bus może być pomylony z car, truck  
    7: [2, 5, 7],  # truck może być pomylony z car, bus
    0: [0, 3],     # person może być pomylony z motorcycle
    3: [0, 3]      # motorcycle może być pomylony z person
}

CROSS_CAMERA_DISTANCE_THRESHOLDS = {
    0: 10,   # person
    2: 35,  # car
    3: 15,   # motorcycle
    5: 35,  # bus
    7: 35   # truck
}

MERGEABLE_CLASS_GROUPS = {
    frozenset([0, 3]): "person_motorcycle",  # person + motorcycle
    frozenset([2, 5, 7]): "vehicle"          # car, bus, truck
}

VEHICLE_CROSS_CLASS_THRESHOLDS = {
    frozenset([2, 5]): 35,    # car + bus 
    frozenset([2, 7]): 35,    # car + truck
    frozenset([5, 7]): 35,    # bus + truck
}

object_trajectories = defaultdict(lambda: deque(maxlen=30))
trajectory_colors = {}

global_object_tracker = {}
next_global_id = 1

DISTANCE_THRESHOLDS = {
    0: 10,   # person
    2: 35,  # car
    3: 15,   # motorcycle
    5: 35,  # bus
    7: 35   # truck
}

# Konfiguracje dla każdego wideo
video_configs = [
    {
        'name': 'Carla1',
        'input_video': f"{data_dir}/output_video0.mp4",
        'output_detect_filename': f"{data_dir}/output_detect_carla1_tracked.mp4",
        'punkty_na_obrazie': np.array([
        (473, 207), (641, 166), (668, 170), (503, 214), 
        (872, 169), (866, 173), (858, 175), (851, 179), 
        (844, 182), (837, 186), (829, 190), (821, 195), 
        (812, 199), (803, 203), (794, 208), (781, 216), 
        (771, 220), (759, 227), (748, 233), (733, 240), 
        (719, 247), (959, 195), (952, 202), 
        (932, 220), (921, 229), (893, 256), (878, 268)
        ], dtype=np.float32),
        'dst_pts_mapa': np.array([
            ((0+offsetX)*s, (0+offsetY)*s), # Bazowy Punkt
            ((298+offsetX)*s, (2+offsetY)*s), #1
            ((296+offsetX)*s, (47+offsetY)*s), #2
            ((0+offsetX)*s, (46+offsetY)*s), #3
            ((450+offsetX)*s, (266+offsetY)*s), #4
            ((422+offsetX)*s, (268+offsetY)*s), #5
            ((395+offsetX)*s, (267+offsetY)*s), #6
            ((367+offsetX)*s, (268+offsetY)*s), #7
            ((340+offsetX)*s, (267+offsetY)*s), #8 
            ((312+offsetX)*s, (268+offsetY)*s), #9
            ((285+offsetX)*s, (267+offsetY)*s), #10
            ((257+offsetX)*s, (268+offsetY)*s), #11
            ((230+offsetX)*s, (269+offsetY)*s), #12
            ((176+offsetX)*s, (269+offsetY)*s), #13
            ((147+offsetX)*s, (270+offsetY)*s), #14
            ((121+offsetX)*s, (270+offsetY)*s), #15
            ((93+offsetX)*s, (271+offsetY)*s), #16
            ((66+offsetX)*s, (271+offsetY)*s), #17
            ((66+offsetX)*s, (271+offsetY)*s), #18
            ((37+offsetX)*s, (270+offsetY)*s), #19
            ((11+offsetX)*s, (271+offsetY)*s), #20
            ((334+offsetX)*s, (395+offsetY)*s), #26
            ((296+offsetX)*s, (394+offsetY)*s), #27
            ((201+offsetX)*s, (394+offsetY)*s), #28
            ((163+offsetX)*s, (394+offsetY)*s), #29
            ((67+offsetX)*s, (394+offsetY)*s), #30
            ((29+offsetX)*s, (393+offsetY)*s) #31
        ], dtype=np.float32),
        'mask_polygon': np.array([
            (0, 343), (75, 330), (307, 274), (387, 231),
            (349, 117), (710, 149), (931, 85),
            (1147, 101), (1280, 582), (1280, 720),
            (0, 720)
        ], dtype=np.int32),
        'H': None, 'cap': None, 'writer_detect': None, 'total_frames': 0, 'fps': 0, 'mask': None
    },
    {
        'name': 'Carla2',
        'input_video': f"{data_dir}/output_video1.mp4",
        'output_detect_filename': f"{data_dir}/output_detect_carla2_tracked.mp4",
        'punkty_na_obrazie': np.array([
        (535, 238), (675, 267), (649, 272), (516, 242), 
        (568, 337), (543, 327), (528, 322), (511, 315), 
        (496, 308), (479, 302), (467, 299), (453, 293), 
        (441, 289), (429, 284), (419, 281), (407, 277), 
        (400, 274), (388, 269), (380, 266), (370, 263), 
        (360, 260), (343, 334), (330, 324), 
        (299, 302), (288, 295), (266, 281), (258, 276)
        ], dtype=np.float32),
        'dst_pts_mapa': np.array([
            ((0+offsetX)*s, (0+offsetY)*s), # Bazowy Punkt
            ((298+offsetX)*s, (2+offsetY)*s), #1
            ((296+offsetX)*s, (47+offsetY)*s), #2
            ((0+offsetX)*s, (46+offsetY)*s), #3
            ((450+offsetX)*s, (266+offsetY)*s), #4
            ((422+offsetX)*s, (268+offsetY)*s), #5
            ((395+offsetX)*s, (267+offsetY)*s), #6
            ((367+offsetX)*s, (268+offsetY)*s), #7
            ((340+offsetX)*s, (267+offsetY)*s), #8 
            ((312+offsetX)*s, (268+offsetY)*s), #9
            ((285+offsetX)*s, (267+offsetY)*s), #10
            ((257+offsetX)*s, (268+offsetY)*s), #11
            ((230+offsetX)*s, (269+offsetY)*s), #12
            ((176+offsetX)*s, (269+offsetY)*s), #13
            ((147+offsetX)*s, (270+offsetY)*s), #14
            ((121+offsetX)*s, (270+offsetY)*s), #15
            ((93+offsetX)*s, (271+offsetY)*s), #16
            ((66+offsetX)*s, (271+offsetY)*s), #17
            ((66+offsetX)*s, (271+offsetY)*s), #18
            ((37+offsetX)*s, (270+offsetY)*s), #19
            ((11+offsetX)*s, (271+offsetY)*s), #20
            ((334+offsetX)*s, (395+offsetY)*s), #26
            ((296+offsetX)*s, (394+offsetY)*s), #27
            ((201+offsetX)*s, (394+offsetY)*s), #28
            ((163+offsetX)*s, (394+offsetY)*s), #29
            ((67+offsetX)*s, (394+offsetY)*s), #30
            ((29+offsetX)*s, (393+offsetY)*s) #31
        ], dtype=np.float32),
        'mask_polygon': np.array([
        (15, 503), (50, 235), (195, 207), (403, 181),
        (416, 211), (438, 225), (537, 223), (836, 195),
        (846, 254), (844, 283), (1280, 350), (1280, 720),
        (0, 720)
        ], dtype=np.int32),
        'H': None, 'cap': None, 'writer_detect': None, 'total_frames': 0, 'fps': 0, 'mask': None
    },
    {
        'name': 'Carla3',
        'input_video': f"{data_dir}/output_video2.mp4",
        'output_detect_filename': f"{data_dir}/output_detect_carla3_tracked.mp4",
        'punkty_na_obrazie': np.array([
        (436, 191), (776, 192), (785, 203), (425, 203), 
        (1170, 304), (1115, 305), (1064, 304), (1008, 304), 
        (959, 305), (905, 305), (852, 306), (799, 307), 
        (745, 306), (692, 307), (640, 308), (584, 308), 
        (531, 308), (476, 309), (423, 310), (358, 310), 
        (314, 310), (1111, 448), (1000, 448), 
        (718, 448), (605, 448), (322, 450), (211, 450)
        ], dtype=np.float32),
        'dst_pts_mapa': np.array([
            ((0+offsetX)*s, (0+offsetY)*s), # Bazowy Punkt
            ((298+offsetX)*s, (2+offsetY)*s), #1
            ((296+offsetX)*s, (47+offsetY)*s), #2
            ((0+offsetX)*s, (46+offsetY)*s), #3
            ((450+offsetX)*s, (266+offsetY)*s), #4
            ((422+offsetX)*s, (268+offsetY)*s), #5
            ((395+offsetX)*s, (267+offsetY)*s), #6
            ((367+offsetX)*s, (268+offsetY)*s), #7
            ((340+offsetX)*s, (267+offsetY)*s), #8 
            ((312+offsetX)*s, (268+offsetY)*s), #9
            ((285+offsetX)*s, (267+offsetY)*s), #10
            ((257+offsetX)*s, (268+offsetY)*s), #11
            ((230+offsetX)*s, (269+offsetY)*s), #12
            ((176+offsetX)*s, (269+offsetY)*s), #13
            ((147+offsetX)*s, (270+offsetY)*s), #14
            ((121+offsetX)*s, (270+offsetY)*s), #15
            ((93+offsetX)*s, (271+offsetY)*s), #16
            ((66+offsetX)*s, (271+offsetY)*s), #17
            ((66+offsetX)*s, (271+offsetY)*s), #18
            ((37+offsetX)*s, (270+offsetY)*s), #19
            ((11+offsetX)*s, (271+offsetY)*s), #20
            ((334+offsetX)*s, (395+offsetY)*s), #26
            ((296+offsetX)*s, (394+offsetY)*s), #27
            ((201+offsetX)*s, (394+offsetY)*s), #28
            ((163+offsetX)*s, (394+offsetY)*s), #29
            ((67+offsetX)*s, (394+offsetY)*s), #30
            ((29+offsetX)*s, (393+offsetY)*s) #31
        ], dtype=np.float32),
        'mask_polygon': np.array([
            (0, 202), (126, 210), (305, 190), (392, 166), 
            (557, 69), (705, 65), (815, 137), (898, 136), 
            (995, 182), (1280, 181), (1280, 720), (0, 720)
        ], dtype=np.int32),
        'H': None, 'cap': None, 'writer_detect': None, 'total_frames': 0, 'fps': 0, 'mask': None
    }
]

# Funkcja do transformacji punktów
def transformuj_wspolrzedne(pt, H):
    """
    Transformuje punkt z obrazu do współrzędnych mapy używając homografii
    """
    x, y = float(pt[0]), float(pt[1])
    vec = np.array([x, y, 1.0], dtype=np.float64)
    # Transformacja
    dst = H.astype(np.float64) @ vec
    # Normalizacja współrzędnych homogenicznych
    transformed_x = dst[0] / dst[2]
    transformed_y = dst[1] / dst[2]
    return float(transformed_x), float(transformed_y)
        

def get_object_base_point(box_coords):
    """
    Oblicza punkt bazowy obiektu na podstawie bounding box
    """
    x1, y1, x2, y2 = map(float, box_coords)
    return (x1 + x2) / 2, y2
    
def merge_duplicate_objects(objects_data, distance_thresholds=DISTANCE_THRESHOLDS):
    """
    Łączy duplikaty obiektów na podstawie odległości z uwzględnieniem kompatybilności klas
    """
    if len(objects_data) <= 1:
        return objects_data
    
    merged_objects = []
    processed_indices = set()
    
    for i, obj1 in enumerate(objects_data):
        if i in processed_indices:
            continue
            
        compatible_objects = [i]
        
        for j, obj2 in enumerate(objects_data):
            if j <= i or j in processed_indices:
                continue
                
            class1, class2 = obj1['class_id'], obj2['class_id']
            can_merge = False
            special_vehicle_threshold = None
            
            if class1 == class2:
                can_merge = True
            else:
                for class_group in MERGEABLE_CLASS_GROUPS.keys():
                    if class1 in class_group and class2 in class_group:
                        can_merge = True
                        break
                
                if not can_merge and {class1, class2}.issubset({2, 5, 7}):
                    class_pair = frozenset([class1, class2])
                    if class_pair in VEHICLE_CROSS_CLASS_THRESHOLDS:
                        can_merge = True
                        special_vehicle_threshold = VEHICLE_CROSS_CLASS_THRESHOLDS[class_pair]
            
            if not can_merge:
                continue
            
            # Określ próg odległości
            if obj1['camera'] != obj2['camera']:
                if class1 == class2:
                    distance_threshold = CROSS_CAMERA_DISTANCE_THRESHOLDS.get(class1, 40)
                elif {class1, class2} == {0, 3}:  # person + motorcycle
                    distance_threshold = 25
                elif special_vehicle_threshold is not None:
                    distance_threshold = special_vehicle_threshold * 0.8
                else:
                    distance_threshold = 30
            else:
                # Ta sama kamera
                if class1 == class2:
                    distance_threshold = distance_thresholds.get(class1, 40)
                elif {class1, class2} == {0, 3}:  # person + motorcycle
                    distance_threshold = 15
                elif special_vehicle_threshold is not None:
                    distance_threshold = special_vehicle_threshold
                else:
                    distance_threshold = 25
            
            x1 = float(obj1['map_x']) if hasattr(obj1['map_x'], 'item') else float(obj1['map_x'])
            y1 = float(obj1['map_y']) if hasattr(obj1['map_y'], 'item') else float(obj1['map_y'])
            x2 = float(obj2['map_x']) if hasattr(obj2['map_x'], 'item') else float(obj2['map_x'])
            y2 = float(obj2['map_y']) if hasattr(obj2['map_y'], 'item') else float(obj2['map_y'])
            
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
            if distance < distance_threshold:
                compatible_objects.append(j)
        
        if len(compatible_objects) > 1:
            classes_in_group = {objects_data[idx]['class_id'] for idx in compatible_objects}
            best_idx = compatible_objects[0]
            best_confidence = float(objects_data[best_idx]['confidence']) if hasattr(objects_data[best_idx]['confidence'], 'item') else float(objects_data[best_idx]['confidence'])
            
            if classes_in_group == {0, 3}:  # person + motorcycle
                class_priority = {0: 6, 3: 5} 
            else:
                class_priority = {7: 4, 5: 3, 2: 2, 0: 1, 3: 1}
            
            for idx in compatible_objects[1:]:
                obj = objects_data[idx]
                confidence = float(obj['confidence']) if hasattr(obj['confidence'], 'item') else float(obj['confidence'])
                if classes_in_group == {0, 3}:
                    if (obj['class_id'] == 0 and objects_data[best_idx]['class_id'] == 3) or \
                       (confidence > best_confidence + 0.15):
                        best_idx = idx
                        best_confidence = confidence
                else:
                    if (confidence > best_confidence + 0.2 or 
                        (abs(confidence - best_confidence) < 0.1 and 
                         class_priority.get(obj['class_id'], 0) > class_priority.get(objects_data[best_idx]['class_id'], 0))):
                        best_idx = idx
                        best_confidence = confidence
        
            best_obj = objects_data[best_idx].copy()
            
            total_weight = 0.0
            weighted_x = 0.0
            weighted_y = 0.0
            cameras = set()
            merged_classes = set()
            
            for idx in compatible_objects:
                obj = objects_data[idx]
                
                weight = float(obj['confidence']) if hasattr(obj['confidence'], 'item') else float(obj['confidence'])
                map_x = float(obj['map_x']) if hasattr(obj['map_x'], 'item') else float(obj['map_x'])
                map_y = float(obj['map_y']) if hasattr(obj['map_y'], 'item') else float(obj['map_y'])
                
                weighted_x += map_x * weight
                weighted_y += map_y * weight
                total_weight += weight  
                cameras.add(obj['camera'])
                merged_classes.add(obj['class_id'])
            
            if total_weight > 0:
                best_obj['map_x'] = float(weighted_x / total_weight)
                best_obj['map_y'] = float(weighted_y / total_weight)
            
            best_obj['cameras'] = list(cameras)
            best_obj['camera'] = f"Multi({len(cameras)})"
            best_obj['merged_count'] = len(compatible_objects)
            best_obj['merged_classes'] = list(merged_classes)
            
            if merged_classes == {0, 3}:
                best_obj['class_name'] = "person+moto"
                best_obj['is_person_motorcycle_group'] = True
            elif len(merged_classes) > 1 and merged_classes.issubset({2, 5, 7}):
                vehicle_names = [CLASS_NAMES[cls][:3] for cls in merged_classes]
                best_obj['class_name'] = f"{best_obj['class_name']}+{'+'.join([n for n in vehicle_names if n != best_obj['class_name'][:3]])}"
                best_obj['is_merged_vehicle'] = True
            
            merged_objects.append(best_obj)
            processed_indices.update(compatible_objects)
        else:
            obj = objects_data[i].copy()
            obj['map_x'] = float(obj['map_x']) if hasattr(obj['map_x'], 'item') else float(obj['map_x'])
            obj['map_y'] = float(obj['map_y']) if hasattr(obj['map_y'], 'item') else float(obj['map_y'])
            obj['confidence'] = float(obj['confidence']) if hasattr(obj['confidence'], 'item') else float(obj['confidence'])
            obj['cameras'] = [obj['camera']]
            obj['merged_count'] = 1
            obj['merged_classes'] = [obj['class_id']]
            merged_objects.append(obj)
            processed_indices.add(i)
    
    return merged_objects

def stabilize_object_class(obj_key, new_class_id, new_confidence):
    """
    Stabilizuje klasę obiektu na podstawie historii detekcji
    """
    new_class_id = int(new_class_id) if hasattr(new_class_id, 'item') else int(new_class_id)
    new_confidence = float(new_confidence) if hasattr(new_confidence, 'item') else float(new_confidence)
    
    if obj_key not in global_object_tracker:
        return new_class_id
    
    obj_data = global_object_tracker[obj_key]
    
    if 'class_history' not in obj_data:
        obj_data['class_history'] = defaultdict(float)
        obj_data['class_history'][new_class_id] = new_confidence
        return new_class_id

    obj_data['class_history'][new_class_id] += new_confidence
    
    best_class = max(obj_data['class_history'].items(), key=lambda x: x[1])
    
    current_best_score = obj_data['class_history'].get(obj_data.get('stable_class_id', best_class[0]), 0.0)
    
    if best_class[1] > current_best_score * 1.5:
        obj_data['stable_class_id'] = best_class[0]
        return best_class[0]
    
    return obj_data.get('stable_class_id', new_class_id)

def main():
    try:
        #model = YOLO(model_path)
        print(f"Załadowano model YOLO: {model_path}")
    except Exception as e:
        print(f"Błąd podczas ładowania modelu YOLO: {e}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    bird_size = (homografiaX, homografiaY)

    # Inicjalizacja konfiguracji wideo
    max_total_frames = 0
    successful_configs = []
    
    for config in video_configs:
        print(f"\nInicjalizacja {config['name']}...")
        config['model'] = YOLO(model_path)
        cap = cv2.VideoCapture(config['input_video'])
        if not cap.isOpened():
            print(f"Nie można otworzyć: {config['input_video']}")
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Tworzenie maski na podstawie wielokąta
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [config['mask_polygon']], 255)
        config['mask'] = mask
        
        config.update({
            'cap': cap,
            'writer_detect': cv2.VideoWriter(config['output_detect_filename'], fourcc, fps, (w, h)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': fps
        })
        
        max_total_frames = max(max_total_frames, config['total_frames'])
        print(f"  Rozdzielczość: {w}x{h}, FPS: {fps}, Klatki: {config['total_frames']}")

        src_pts_np = config['punkty_na_obrazie']
        dst_pts_np = config['dst_pts_mapa']

        H = cv2.findHomography(src_pts_np, dst_pts_np, cv2.RANSAC)
        config['H'] = H[0]

        successful_configs.append(config)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    active_video_configs = successful_configs
    output_fps = active_video_configs[0]['fps'] if active_video_configs and active_video_configs[0]['fps'] > 0 else 30
    out_bird_combined = cv2.VideoWriter(f"{data_dir}/output_birdview_COMBINED_tracked.mp4", fourcc, output_fps, bird_size)

    if max_total_frames == 0:
        print("Brak klatek do przetworzenia.")
        return

    # Główna pętla przetwarzania
    accumulator = np.zeros((homografiaY, homografiaX, 3), dtype=np.float32)
    coverage = np.zeros((homografiaY, homografiaX), dtype=np.uint16)

    global object_trajectories, trajectory_colors
    global global_object_tracker, next_global_id
    global_object_tracker = {}
    next_global_id = 1
    object_trajectories = defaultdict(lambda: deque(maxlen=30))
    trajectory_colors = {}

    pbar = tqdm(total=max_total_frames, desc="Przetwarzanie klatek")

    for frame_idx in range(max_total_frames):
        active_configs_this_frame = []
        original_frames_this_frame = []
        yolo_results_this_frame = []
        
        frame_objects_detected = 0
        frame_objects_transformed = 0

        # Wczytywanie klatek z wszystkich kamer
        for cfg_idx, cfg in enumerate(active_video_configs):
            if frame_idx < cfg['total_frames']:
                ret, frame = cfg['cap'].read()
                if ret:
                    active_configs_this_frame.append(cfg)
                    original_frames_this_frame.append(frame.copy())
        
        # Detekcja i śledzenie obiektów
        for cfg, orig_frame in zip(active_configs_this_frame, original_frames_this_frame):
            if orig_frame is not None:
                results = cfg['model'].track(
                    orig_frame, 
                    conf=conf_threshold, 
                    persist=True, 
                    verbose=False, 
                    classes=TARGET_CLASS_IDS,
                    iou=0.3,
                    half=False,
                    device='gpu',
                    imgsz=1280,
                    max_det=100,
                    agnostic_nms=False,
                    retina_masks=False,
                    tracker="bytetrack.yaml" 
                )
                
                yolo_results_this_frame.append(results)
                
                annotated_frame = results[0].plot(
                    line_width=2,
                    font_size=1.0,
                    pil=False,
                    labels=True,
                    boxes=True,
                    conf=True
                )
                cv2.imshow(f"Tracked Objects {cfg['name']}", annotated_frame)
                cfg['writer_detect'].write(annotated_frame)

        # Tworzenie widoku z lotu ptaka
        accumulator.fill(0)
        coverage.fill(0)

        for cfg, orig_frame in zip(active_configs_this_frame, original_frames_this_frame):
            if orig_frame is not None and cfg['H'] is not None:
                masked_frame = cv2.bitwise_and(orig_frame, orig_frame, mask=cfg['mask'])
                bird_view_frame = cv2.warpPerspective(masked_frame, cfg['H'], bird_size).astype(np.float32)
                gray_bird_view = cv2.cvtColor(bird_view_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                mask = gray_bird_view > 0
                accumulator[mask] += bird_view_frame[mask]
                coverage[mask] += 1

        # Tworzenie połączonego widoku
        combined_bird_eye_view = np.zeros_like(accumulator, dtype=np.uint8)
        valid_coverage_mask = coverage > 0
        combined_bird_eye_view[valid_coverage_mask] = (accumulator[valid_coverage_mask] / coverage[valid_coverage_mask, None]).astype(np.uint8)

        # Transformowanie i rysowanie śledzonych obiektów
        all_frame_objects = []
        
        for cfg_idx, (cfg, yolo_result_list) in enumerate(zip(active_configs_this_frame, yolo_results_this_frame)):
            yolo_result_object = yolo_result_list[0]
            
            if yolo_result_object.boxes.id is None:
                boxes_data = []
                for idx, (box_coords, box_cls) in enumerate(zip(yolo_result_object.boxes.xyxy, yolo_result_object.boxes.cls)):
                    confidence = yolo_result_object.boxes.conf[idx] if hasattr(yolo_result_object.boxes, 'conf') and len(yolo_result_object.boxes.conf) > idx else 1.0
                    
                    # Znajdź najbliższy obiekt z poprzednich klatek
                    base_point = get_object_base_point(box_coords)
                    class_id = int(box_cls)
                    
                    best_match_id = None
                    min_distance = float('inf')
                    
                    for global_id, obj_data in global_object_tracker.items():
                        if (obj_data['class_id'] == class_id and 
                            obj_data['camera'] == cfg['name'] and
                            frame_idx - obj_data['last_frame'] < 10):
                            
                            distance = np.sqrt((base_point[0] - obj_data['last_pos'][0])**2 + 
                                            (base_point[1] - obj_data['last_pos'][1])**2)
                            
                            if distance < DISTANCE_THRESHOLDS.get(class_id, 30) and distance < min_distance:
                                min_distance = distance
                                best_match_id = global_id
                    
                    if best_match_id is not None:
                        track_id = best_match_id
                        global_object_tracker[best_match_id].update({
                            'last_pos': base_point,
                            'last_frame': frame_idx
                        })
                    else:
                        track_id = next_global_id
                        next_global_id += 1
                        global_object_tracker[track_id] = {
                            'class_id': class_id,
                            'camera': cfg['name'],
                            'last_pos': base_point,
                            'last_frame': frame_idx
                        }
                    
                    boxes_data.append((box_coords, box_cls, track_id, confidence))
            else:
                boxes_data = []
                for i in range(len(yolo_result_object.boxes.id)):
                    box_coords = yolo_result_object.boxes.xyxy[i]
                    box_cls = yolo_result_object.boxes.cls[i]
                    track_id = int(yolo_result_object.boxes.id[i].item()) if hasattr(yolo_result_object.boxes.id[i], 'item') else int(yolo_result_object.boxes.id[i])
                    confidence = yolo_result_object.boxes.conf[i] if hasattr(yolo_result_object.boxes, 'conf') and len(yolo_result_object.boxes.conf) > i else 1.0
                    
                    base_point = get_object_base_point(box_coords)
                    class_id = int(box_cls)
                    
                    global_track_id = f"{cfg['name']}_{track_id}"
                    global_object_tracker[global_track_id] = {
                        'class_id': class_id,
                        'camera': cfg['name'],
                        'last_pos': base_point,
                        'last_frame': frame_idx
                    }
                    
                    boxes_data.append((box_coords, box_cls, global_track_id, confidence))
            
            for box_coords, box_cls, track_id_raw, confidence in boxes_data:
                class_id = int(box_cls)
                
                if class_id not in TARGET_CLASS_IDS:
                    continue
                
                frame_objects_detected += 1
                
                if isinstance(track_id_raw, str):
                    track_id = track_id_raw
                    display_id = f"D{frame_objects_detected}"
                else:
                    track_id = int(track_id_raw.item()) if hasattr(track_id_raw, 'item') else int(track_id_raw)
                    display_id = str(track_id)
                
                obj_key = f"{cfg['name']}_{track_id}"
                stable_class_id = stabilize_object_class(obj_key, class_id, confidence)
                
                class_name = CLASS_NAMES.get(stable_class_id, f"class_{stable_class_id}")
                base_point = get_object_base_point(box_coords)
                
                pt_transformed = transformuj_wspolrzedne(base_point, cfg['H'])
                
                if pt_transformed[0] is not None and pt_transformed[1] is not None:
                    pt_on_map = (int(round(pt_transformed[0])), int(round(pt_transformed[1])))
                    
                    margin = 50
                    if (-margin <= pt_on_map[0] < homografiaX + margin and 
                        -margin <= pt_on_map[1] < homografiaY + margin):
                        
                        safe_x = max(5, min(homografiaX - 5, pt_on_map[0]))
                        safe_y = max(5, min(homografiaY - 5, pt_on_map[1]))
                        
                        all_frame_objects.append({
                            'map_x': float(safe_x),
                            'map_y': float(safe_y),
                            'class_id': int(stable_class_id),
                            'class_name': class_name,
                            'track_id': track_id,
                            'display_id': display_id,
                            'confidence': float(confidence.item()) if hasattr(confidence, 'item') else float(confidence),
                            'camera': cfg['name'],
                            'original_class_id': int(class_id)
                        })
        
        merged_objects = merge_duplicate_objects(all_frame_objects)

        current_frame_objects = set()
        for obj in merged_objects:
            obj_key = f"{obj['camera']}_{obj['track_id']}_{obj['class_id']}"
            current_frame_objects.add(obj_key)
            
            object_trajectories[obj_key].append((obj['map_x'], obj['map_y'], frame_idx))
            
            if obj_key not in trajectory_colors:
                import hashlib
                hash_obj = hashlib.md5(obj_key.encode())
                hash_hex = hash_obj.hexdigest()
                r = int(hash_hex[0:2], 16)
                g = int(hash_hex[2:4], 16) 
                b = int(hash_hex[4:6], 16)
                if r + g + b < 300:
                    r = min(255, r + 100)
                    g = min(255, g + 100)
                    b = min(255, b + 100)
                trajectory_colors[obj_key] = (int(b), int(g), int(r))
        
        # Rysowanie połączonych obiektów
        for obj in merged_objects:
            safe_pt = (int(obj['map_x']), int(obj['map_y']))
            frame_objects_transformed += 1
            
            # Rysowanie punktu z animacją
            pulse = int(5 + 3 * abs(np.sin(frame_idx * 0.1)))
            
            color_map = {
                0: (255, 100, 100),   # person
                2: (100, 255, 100),   # car
                3: (100, 100, 255),   # motorcycle
                5: (255, 255, 100),   # bus
                7: (255, 100, 255)    # truck
            }
            if obj.get('is_person_motorcycle_group', False):
                color = (255, 150, 50)
            else:
                color = color_map.get(obj['class_id'], (255, 255, 255))
            
            # Większe koło dla obiektów z wielu kamer
            extra_size = 2 if 'cameras' in obj and len(obj['cameras']) > 1 else 0
            
            cv2.circle(combined_bird_eye_view, safe_pt, pulse + 2 + extra_size, (0, 0, 0), -1)
            cv2.circle(combined_bird_eye_view, safe_pt, pulse + extra_size, color, -1)
            cv2.circle(combined_bird_eye_view, safe_pt, pulse + 1 + extra_size, (255, 255, 255), 1)
            
            # Tekst z informacją o kamerach i stabilizacji
            if 'cameras' in obj and len(obj['cameras']) > 1:
                stability_indicator = "★" if obj.get('original_class_id') != obj['class_id'] else ""
                
                # Specjalna obsługa dla grup
                if obj.get('is_person_motorcycle_group', False):
                    text = f"{obj['display_id']}:P+M[{len(obj['cameras'])}]{stability_indicator}"
                else:
                    text = f"{obj['display_id']}:{obj['class_name'][:3]}[{len(obj['cameras'])}]{stability_indicator}"
            else:
                stability_indicator = "★" if obj.get('original_class_id') != obj['class_id'] else ""
                
                if obj.get('is_person_motorcycle_group', False):
                    text = f"{obj['display_id']}:P+M{stability_indicator}"
                else:
                    text = f"{obj['display_id']}:{obj['class_name'][:3]}{stability_indicator}"
            
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = max(0, min(homografiaX - text_size[0], safe_pt[0] - text_size[0] // 2))
            text_y = max(text_size[1], min(homografiaY - 5, safe_pt[1] - 15))
            
            # Tło dla tekstu
            cv2.rectangle(combined_bird_eye_view, 
                        (text_x - 2, text_y - text_size[1] - 2),
                        (text_x + text_size[0] + 2, text_y + 2),
                        (0, 0, 0), -1)

            # Sam tekst
            cv2.putText(combined_bird_eye_view, text, (text_x, text_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Rysowanie trajektorii
        for obj_key, trajectory in object_trajectories.items():
            if len(trajectory) > 1:
                color = trajectory_colors.get(obj_key, (128, 128, 128))
                
                # Rysuj linie między kolejnymi punktami
                points = list(trajectory)
                for i in range(1, len(points)):
                    pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                    pt2 = (int(points[i][0]), int(points[i][1]))
                    
                    # Sprawdź czy punkty są w granicach obrazu
                    if (0 <= pt1[0] < homografiaX and 0 <= pt1[1] < homografiaY and
                        0 <= pt2[0] < homografiaX and 0 <= pt2[1] < homografiaY):
                        
                        thickness = 2
                        
                        cv2.line(combined_bird_eye_view, pt1, pt2, color, thickness)
                
                # Dodaj punkt końcowy trajektorii
                if len(points) > 0:
                    last_pt = (int(points[-1][0]), int(points[-1][1]))
                    if 0 <= last_pt[0] < homografiaX and 0 <= last_pt[1] < homografiaY:
                        cv2.circle(combined_bird_eye_view, last_pt, 3, color, -1)

        keys_to_remove = []
        for obj_key in list(object_trajectories.keys()):
            if obj_key not in current_frame_objects:
                if (len(object_trajectories[obj_key]) > 0 and 
                    frame_idx - object_trajectories[obj_key][-1][2] > 30):
                    keys_to_remove.append(obj_key)
        
        for key in keys_to_remove:
            del object_trajectories[key]
            if key in trajectory_colors:
                del trajectory_colors[key]

        # Zapis i wyświetlanie
        out_bird_combined.write(combined_bird_eye_view)
        cv2.imshow("Combined Bird's-eye View with Tracks", combined_bird_eye_view)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

        if frame_idx % 30 == 0:  # Co 30 klatek czyść stare obiekty
            keys_to_remove = []
            for global_id, obj_data in global_object_tracker.items():
                if frame_idx - obj_data['last_frame'] > 30:
                    keys_to_remove.append(global_id)
            
            for key in keys_to_remove:
                del global_object_tracker[key]

        pbar.update(1)

    # Cleanup
    for cfg in active_video_configs:
        if cfg.get('cap'): 
            cfg['cap'].release()
        if cfg.get('writer_detect'): 
            cfg['writer_detect'].release()
    
    if out_bird_combined.isOpened(): 
        out_bird_combined.release()
    cv2.destroyAllWindows()
    pbar.close()
    
    print("Zakończono przetwarzanie.")

if __name__ == '__main__':
    main()