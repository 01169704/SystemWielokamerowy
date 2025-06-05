import sys
import os
import time
import random
import cv2
import numpy as np
from PIL import Image

# Add CARLA PythonAPI to path
try:
    sys.path.append(r"E:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.14-py3.7-win-amd64.egg")
    import carla
except ImportError:
    raise RuntimeError("Could not import CARLA. Check your PythonAPI path")


def setup_carla_world():
    """Połączenie z serwerem CARLA i przygotowanie świata"""
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    return client, world


def spawn_vehicles(world, count=15):
    """Losowe pojazdy w świecie i uruchomienie autopilota"""
    blueprints = world.get_blueprint_library().filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    vehicles = []
    for i in range(min(count, len(spawn_points))):
        vehicle_bp = random.choice(blueprints)
        try:
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[i])
            if vehicle:
                vehicle.set_autopilot(True)  # Uruchomienie autopilota
                vehicles.append(vehicle)
                print(f"Spawned vehicle: {vehicle.type_id}")
        except Exception as e:
            print(f"Failed to spawn vehicle: {e}")
    return vehicles


def spawn_walkers(world, client, count=10):
    """Tworzenie pieszych z kontrolerami AI"""
    walker_blueprints = world.get_blueprint_library().filter('walker.pedestrian.*')
    spawn_points = []

    for _ in range(count):
        loc = world.get_random_location_from_navigation()
        if loc:
            spawn_points.append(carla.Transform(loc))

    walkers = []
    controllers = []
    for transform in spawn_points:
        walker_bp = random.choice(walker_blueprints)
        try:
            walker = world.try_spawn_actor(walker_bp, transform)
            if walker:
                controller_bp = world.get_blueprint_library().find('controller.ai.walker')
                controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
                controller.start()
                controller.go_to_location(world.get_random_location_from_navigation())
                controller.set_max_speed(1 + random.random())  # prędkość 1–2 m/s
                walkers.append(walker)
                controllers.append(controller)
        except Exception as e:
            print(f"Failed to spawn walker: {e}")
    return walkers, controllers


def setup_cameras(world, positions):
    """Tworzenie zestawu kamer"""
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1280')
    camera_bp.set_attribute('image_size_y', '720')
    camera_bp.set_attribute('fov', '90')

    cameras = []
    for idx, (location, rotation) in enumerate(positions):
        transform = carla.Transform(location, rotation)
        try:
            camera = world.spawn_actor(camera_bp, transform)
            cameras.append((camera, f'_cam{idx}'))
            print(f"Camera {idx} created at {location}")
        except Exception as e:
            print(f"Failed to spawn camera {idx}: {e}")
    return cameras


def process_image(camera_id, output_dir):
    """Zwraca funkcję zapisującą zdjęcia z kamery"""
    os.makedirs(output_dir, exist_ok=True)

    def callback(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        bgr_image = array[:, :, :3]
        timestamp = int(time.time() * 1000)
        filename = os.path.join(output_dir, f"{camera_id}_{timestamp}.png")
        cv2.imwrite(filename, bgr_image)
        print(f"[{camera_id}] Saved image: {filename}")

    return callback


def main():
    vehicles = []
    walkers = []
    controllers = []
    cameras = []

    try:
        client, world = setup_carla_world()

        vehicles = spawn_vehicles(world, count=30)
        walkers, controllers = spawn_walkers(world, client, count=20)

        camera_positions = [
             (carla.Location(x=-63, y=120, z=7), carla.Rotation(pitch=-30, yaw=20)),
            (carla.Location(x=-47, y=148, z=7), carla.Rotation(pitch=-30, yaw=-90)),
            (carla.Location(x=-32, y=118, z=7), carla.Rotation(pitch=-30, yaw=150))

        ]
        cameras = setup_cameras(world, camera_positions)

        for idx, (camera, folder) in enumerate(cameras):
            camera.listen(process_image(f"cam{idx}", folder))

        print("Monitoring skrzyżowania rozpoczęty. Naciśnij Ctrl+C, aby zakończyć.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nPrzerwano przez użytkownika")
    except Exception as e:
        print(f"Błąd: {e}")
    finally:
        print("Czyszczenie świata...")

        for camera, _ in cameras:
            if camera.is_alive:
                camera.stop()
                camera.destroy()

        for vehicle in vehicles:
            if vehicle.is_alive:
                vehicle.destroy()

        for w in walkers + controllers:
            if w.is_alive:
                w.destroy()

        print("Zakończono symulację.")


if __name__ == '__main__':
    main()
