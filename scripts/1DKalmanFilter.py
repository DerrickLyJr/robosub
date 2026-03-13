import numpy as np
import matplotlib.pyplot as plt

# 1. Setup Simulation Parameters
total_time = 100  # seconds
dt = 0.1          # 10Hz processing
steps = int(total_time / dt)

# "True" Depth (Constant dive at 0.1 m/s)
true_depth = np.linspace(0, 10, steps) 

# 2. Initialize Kalman Filter Variables
depth = 0.0       # Initial depth estimate
Variance = 1.0       # Initial uncertainty (Variance)
imuNoise = 0.01     # Process Noise (How much the IMU drifts)
pressureNoise = 0.5       # Measurement Noise (How shaky the Pressure sensor is)

# Storage for plotting
estimates = []
observations = []

# 3. The Fusion Loop
for i in range(steps):
    # --- STEP 1: PREDICT (Using "Fake IMU") ---
    # We think we moved 0.01m since the last 0.1s step
    imu_move = 0.01 + np.random.normal(0, 0.005) # Add small drift
    depth = depth + imu_move
    Variance = Variance + imuNoise  # Uncertainty grows because we are just guessing
    
    # --- STEP 2: UPDATE (Using "Fake Pressure Sensor") ---
    # In this practice, we only get a pressure reading every 1 second (10 steps)
    if i % 10 == 0:
        pressureSensorNoise = true_depth[i] + np.random.normal(0, np.sqrt(pressureNoise)) # Fake noisy sensor
        observations.append((i, pressureSensorNoise))
        
        # Calculate Kalman Gain (How much do we trust this sensor?)
        K = Variance / (Variance + pressureNoise) # we are k% certain about our prediction, and (1-K)% certain about the sensor reading
        
        # Correct our estimate
        depth = depth + K * (pressureSensorNoise - depth)
        
        # Update our uncertainty (It shrinks because we just checked a sensor!)
        Variance = (1 - K) * Variance # We are more certain now since we just got a measurement
        
    estimates.append(depth)

# 4. Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(true_depth, label='True Depth (Ground Truth)', color='black', linestyle='--')
plt.plot(estimates, label='Kalman Filter Estimate', color='blue', linewidth=2)
obs_idx, obs_val = zip(*observations)
plt.scatter(obs_idx, obs_val, color='red', s=10, label='Noisy Pressure Readings')

plt.title("Sensor Fusion: IMU Prediction + Pressure Correction")
plt.xlabel("Time Steps (0.1s)")
plt.ylabel("Depth (Meters)")
plt.legend()
plt.show()