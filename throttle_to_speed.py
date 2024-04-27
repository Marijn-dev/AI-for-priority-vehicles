throttles = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
distances = [6.104819476604462e-07, 5.634315311908722e-05, 8.391216397285461e-07, 7.350929081439972e-05, 1.5388087034225464, 1.8643467426300049, 2.5804202556610107, 4.615363597869873, 5.702938079833984, 6.893794059753418, 8.280390739440918, 9.894770622253418, 22.578275680541992, 49.22041320800781, 63.310157775878906, 77.92605590820312, 89.9015884399414, 100.43115234375, 109.56205749511719, 119.4863052368164]

# Compute average speeds (distance / 5 seconds)
average_speeds = [distance / 5 for distance in distances]

# Now fit these to a model; a polynomial fit could be attempted here
import numpy as np
import matplotlib.pyplot as plt

# Fit a polynomial curve
z = np.polyfit(throttles, average_speeds, 3)  # Using a cubic polynomial
p = np.poly1d(z)

# # Plotting
# xp = np.linspace(0, 1, 100)
# _ = plt.plot(throttles, average_speeds, '.', xp, p(xp), '-')
# plt.xlabel('Throttle')
# plt.ylabel('Speed (m/s)')
# plt.title('Throttle vs Speed')
# plt.grid(True)
# plt.show()

# Predict speed for a given throttle
throttle_value = 0.9
predicted_speed = p(throttle_value)
print(f"Predicted speed for throttle {throttle_value}: {predicted_speed} m/s")
print(f"Predicted speed for throttle {throttle_value}: {predicted_speed*3.6} km/h")
