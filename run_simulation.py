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
buoyancy_force = mass * 9.81      # robot paani mein tairta hai
drag_coefficient = 8.0            # paani ki rukawat

# 3. Jagahein (x, y, z) meters mein
home = np.array([0.0, 0.0, 1.0])
goal = np.array([3.0, 2.0, 0.6])
waypoints = [goal, home]          # pehle goal, phir home
wp_index = 0
mission_done = False

# 4. Controller settings
kp = 15.0
kd = 17.0
max_force = 15.0

# 5. Battery
battery_pct = 100.0
low_battery_limit = 30.0
returning = False

# 6. Graph ke liye data
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

        # Controller
        target = waypoints[wp_index]
        error = target - position
        drag = -drag_coefficient * velocity
        control = np.clip(kp * error - kd * velocity, -max_force, max_force)

        # Forces lagao
        data.xfrc_applied[auv_id, :] = 0
        data.xfrc_applied[auv_id, :3] = drag + control
        data.xfrc_applied[auv_id, 2] += buoyancy_force

        mujoco.mj_step(model, data)

        # Battery kam karo
        battery_pct -= 0.002 + 0.0005 * np.linalg.norm(control)

        # Battery kam? Seedha home jao
        if battery_pct < low_battery_limit and not returning:
            print("LOW BATTERY! Returning home...")
            wp_index = len(waypoints) - 1
            returning = True

        # Graph ka data save karo
        times.append(data.time)
        z_values.append(position[2])
        battery_values.append(battery_pct)

        # Target par pohanche?
        if np.linalg.norm(error) < 0.10:
            if wp_index < len(waypoints) - 1:
                print("Goal reached! Going back home...")
                wp_index += 1
            elif not mission_done:
                print("Mission complete: robot is back home")
                mission_done = True

        # Terminal par status
        step += 1
        if step % 100 == 0:
            print("Pos:", position.round(2),
                  "| Battery:", round(battery_pct, 1), "%")

        viewer.sync()

# 7. Viewer band hone ke baad graph banao
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