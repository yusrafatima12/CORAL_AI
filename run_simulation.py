import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

# 1. World load karo
model = mujoco.MjModel.from_xml_path("simulation/underwater_world.xml")
data = mujoco.MjData(model)
auv_id = model.body("auv").id

# 2. Physics
mass = model.body_mass[auv_id]
buoyancy_force = mass * 9.81
drag_coefficient = 8.0

# 3. Jagahein
home = np.array([0.0, 0.0, 1.0])
goal = np.array([3.0, 2.0, 0.6])
waypoints = [goal, home]
names = ["goal", "home"]          # har target ka naam
wp_index = 0
mission_done = False

# 4. Controller
kp = 15.0
kd = 17.0
max_force = 15.0

# 5. Battery
battery_pct = 100.0
low_battery_limit = 30.0
returning = False

# 6. Sonar settings
sonar_range = 1.0         # 1 meter tak dekhta hai
detour_distance = 1.0     # patthar se kitna side mein hatna hai
obstacle_warned = False

# 7. Graph data
times = []
z_values = []
battery_values = []

print("Mission start")

with mujoco.viewer.launch_passive(model, data) as viewer:

    # 10 second ruko, camera set karo
    wait_start = time.time()
    while time.time() - wait_start < 10:
        viewer.sync()
        time.sleep(0.05)

    step = 0
    while viewer.is_running():
        position = data.qpos[:3]
        velocity = data.qvel[:3]

        target = waypoints[wp_index]
        error = target - position

        # --- SONAR: target ki taraf ray bhejo ---
        direction = error / (np.linalg.norm(error) + 1e-6)
        geomid = np.zeros(1, dtype=np.int32)
        sonar_dist = mujoco.mj_ray(
            model, data, position.copy(), direction, None, 1, auv_id, geomid
        )

        # Patthar mila? To detour point banao
        if (0 < sonar_dist < sonar_range
                and not obstacle_warned
                and names[wp_index] != "detour"):
            hit_point = position + direction * sonar_dist
            side = np.array([-direction[1], direction[0], 0.0])  # left side
            detour = hit_point + side * detour_distance
            detour[2] = position[2]
            waypoints.insert(wp_index, detour)
            names.insert(wp_index, "detour")
            obstacle_warned = True
            print("SONAR: obstacle detected! Going around it...")
            continue   # naye target ke saath loop dobara shuru

        # Controller
        drag = -drag_coefficient * velocity
        control = np.clip(kp * error - kd * velocity, -max_force, max_force)

        # Forces lagao
        data.xfrc_applied[auv_id, :] = 0
        data.xfrc_applied[auv_id, :3] = drag + control
        data.xfrc_applied[auv_id, 2] += buoyancy_force

        mujoco.mj_step(model, data)

        # Battery
        battery_pct -= 0.002 + 0.0005 * np.linalg.norm(control)
        if battery_pct < low_battery_limit and not returning:
            print("LOW BATTERY! Returning home...")
            wp_index = len(waypoints) - 1
            returning = True

        # Graph data
        times.append(data.time)
        z_values.append(position[2])
        battery_values.append(battery_pct)

        # Target par pohanche?
        if np.linalg.norm(error) < 0.10:
            if names[wp_index] == "detour":
                print("Obstacle passed, continuing...")
                wp_index += 1
                obstacle_warned = False
            elif names[wp_index] == "goal":
                print("Goal reached! Going back home...")
                wp_index += 1
                obstacle_warned = False
            elif not mission_done:
                print("Mission complete: robot is back home")
                mission_done = True

        step += 1
        if step % 100 == 0:
            print("Pos:", position.round(2),
                  "| Target:", names[wp_index],
                  "| Battery:", round(battery_pct, 1), "%")

        viewer.sync()

# 8. Graphs
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(times, z_values)
plt.xlabel("Time (s)")
plt.ylabel("Z position (m)")
plt.title("Depth vs Time")

plt.subplot(1, 2, 2)
plt.plot(times, battery_values, color="green")
plt.xlabel("Time (s)")
plt.ylabel("Battery (%)")
plt.title("Battery vs Time")

plt.tight_layout()
plt.show()