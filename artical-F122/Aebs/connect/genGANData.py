import carla
import random
import time
import mss
import cv2
import numpy as np
import os
import csv

# ========== Parameters ==========
OUT_DIR = "./Aebs/carla_data/"
NUM_SAMPLES = 400
Z_OFFSET = 2.0
SCREENSHOT_REGION = {'top': 50, 'left': 40, 'width': 2520, 'height': 1550}
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 640

# Create output directory
os.makedirs(OUT_DIR, exist_ok=True)

# Generate fixed 400 sampling points (order: near to far)
distances_near = np.linspace(5.0, 10.0, 200)
distances_far = np.linspace(10.0, 16.0, 200)
DISTANCE_SAMPLES = np.concatenate((distances_near, distances_far))

# Write CSV header
CSV_PATH = os.path.join(OUT_DIR, "labels.csv")
with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "distance_m"])

# ========== 1. Connect Client and Load Map ==========
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
available_maps = client.get_available_maps()
map_to_load = available_maps[5]
world = client.load_world(map_to_load)
print(f"Map '{map_to_load}' loaded successfully.")

# ========== 2. Set Time to 9:00 AM ==========
world.set_weather(carla.WeatherParameters(
    cloudiness=0.0,
    precipitation=0.0,
    precipitation_deposits=0.0,
    wind_intensity=0.0,
    fog_density=0.0,
    wetness=0.0,
    sun_altitude_angle=60.0,
    sun_azimuth_angle=90.0
))
print("Time set to 9:00 AM.")

# ========== 3. Set Observer Starting Point ==========
spectator = world.get_spectator()
TARGET_LOCATION = carla.Location(x=42.846504, y=-193.132416, z=0.275307)
FIXED_ROTATION = carla.Rotation(pitch=0, yaw=1.5, roll=0)
spectator_transform = carla.Transform(TARGET_LOCATION + carla.Location(z=Z_OFFSET), FIXED_ROTATION)
spectator.set_transform(spectator_transform)
print("Observer initialized.")

# ========== 4. Spawn Vehicle ==========
blueprint_library = world.get_blueprint_library()
car_bp = random.choice(blueprint_library.filter('vehicle.tesla.model3'))
car_bp.set_attribute('color', '255,0,0')
vehicle_transform = carla.Transform(TARGET_LOCATION, FIXED_ROTATION)
vehicle = world.try_spawn_actor(car_bp, vehicle_transform)

if vehicle is None:
    raise RuntimeError("Vehicle spawn failed. Please check if the map coordinates are valid.")
print("Vehicle spawned. Ready to collect data.")


# ===== Add this section before data collection starts =====
WAIT_SECONDS = 8

print(f"\nScreenshot capture starting in {WAIT_SECONDS} seconds. Please adjust window position...")
for i in range(WAIT_SECONDS, 0, -1):
    print(f"Starting collection in {i} seconds...", end="\r")
    time.sleep(1)
print("\nStarting collection!\n")

# ========== 5. Initialize Screenshot Tool ==========
screenShot = mss.mss()

print(f"Starting sequential image capture. Total samples: {NUM_SAMPLES}")

try:
    for idx, distance in enumerate(DISTANCE_SAMPLES):
        # Get vehicle position
        car_tf = vehicle.get_transform()
        car_loc = car_tf.location
        car_rot = car_tf.rotation

        # Calculate camera position: behind the vehicle
        forward_vector = car_tf.get_forward_vector()
        backward_vector = carla.Location(-forward_vector.x, -forward_vector.y, 0)
        new_location = car_loc + backward_vector * distance + carla.Location(z=Z_OFFSET)

        # Move observer
        spectator.set_transform(carla.Transform(new_location, car_rot))
        time.sleep(0.3)

        # Capture screenshot
        img = np.array(screenShot.grab(SCREENSHOT_REGION))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT))
        filename = f"{idx:04d}.png"
        filepath = os.path.join(OUT_DIR, filename)
        cv2.imwrite(filepath, img)

        # Save to CSV
        with open(CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([filename, distance])

        print(f"[{idx+1}/{NUM_SAMPLES}] Saved {filename}, distance {distance:.2f} meters")

    print("\n All samples collected successfully!")

except KeyboardInterrupt:
    print("\n Collection terminated by user.")

finally:
    if vehicle is not None:
        vehicle.destroy()
        print("Vehicle destroyed.")