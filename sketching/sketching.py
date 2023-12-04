# %%

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
import pandas as pd


img = Image.open("PLS_9771_01_clip.tiff")
downsampling_factor = 0.5  # Adjust this value as needed

# Compute the new size
new_size = (int(img.width * downsampling_factor), int(img.height * downsampling_factor))

# downscale?
# img = img.resize(new_size, Image.ANTIALIAS)
# %%


median_filter = ImageFilter.MedianFilter(size=9)

filtered_img = img
# %%

hsv_img = filtered_img.convert("HSV")

# get h channel (0th matrix)
hue_values = np.array(hsv_img)[:, :, 0].flatten()

# Convert the numpy array to a pandas Series and use value_counts to get the counts
# 255 val because 256 == 0
counts = pd.Series(hue_values).value_counts().reindex(np.arange(255), fill_value=0)

# peaks = counts[counts>img.width*img.height/50]
# print(peaks)

# %%

# from sklearn.neighbors import KernelDensity

# X_plot = np.linspace(0, 255, 255)[:, np.newaxis]


# # Creating a figure
# fig, ax = plt.subplots()

# kde = KernelDensity(kernel="gaussian", bandwidth=5).fit(hue_values.reshape(-1,1))
# # Calculating the log of the probability density function
# log_dens = kde.score_samples(X_plot)


# # Plotting the density curve
# ax.plot(X_plot[:, 0], np.exp(log_dens), color="cornflowerblue", linestyle="-", label="Gaussian kernel density")

# # Set the title, x and y labels of the plot
# ax.set_title("Gaussian Kernel Density")
# ax.set_xlim(0, 255)
# ax.set_ylim(0, 0.001)
# ax.grid(True)
# ax.legend(loc="upper right")

# # Display the plot
# plt.show()

# %%
# Find local maxima
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter1d

smoothed = gaussian_filter1d(counts.values, 5)
peaks, _ = find_peaks(smoothed)

# Extract x and y values of local maxima
local_maxima_y = smoothed[peaks]
local_maxima_x = peaks

# Plot the KDE and the local maxima
plt.plot(counts.values, label="hue counts")
plt.plot(smoothed, label="smoothed hue counts")
plt.plot(local_maxima_x, local_maxima_y, "x", color="r", label="Local Maxima")
plt.xlabel("hue")
plt.ylabel("count")
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(counts.values, label="hue counts")
ax.plot(smoothed, label="smoothed hue counts")
ax.plot(local_maxima_x, local_maxima_y, "x", color="r", label="Local Maxima")
ax.set_xlim(0, 255)
ax.set_ylim(0, 40000)
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(counts.values, label="hue counts")
ax.plot(smoothed, label="smoothed hue counts")
ax.plot(local_maxima_x, local_maxima_y, "x", color="r", label="Local Maxima")
ax.set_xlim(0, 255)
ax.set_ylim(0, 5000)
ax.legend()
plt.show()


data = {val: key for key, val in list(zip(peaks, [counts.values[peak] for peak in peaks]))}
lower_bound = img.width * img.height / 1000
print("lower bound", lower_bound)
outliers = [y for x, y in data.items() if x > lower_bound]
# hue_vals = [data[outlier] for outlier in outliers]

print("Identified hues:", outliers)
# %%

# data = {val: key for key, val in list(zip(peaks,[counts.values[peak] for peak in peaks]))}
# Q1 = np.percentile(list(data.keys()), 15)
# Q3 = np.percentile(list(data.keys()), 85)

# # Calculate the interquartile range (IQR)
# IQR = Q3 - Q1

# # Define the lower and upper bounds to identify outliers
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR

# # Identify outliers
# outliers = [x for x in data if x < lower_bound or x > upper_bound]
# hue_vals = [data[outlier] for outlier in outliers]

# print("Identified outliers:", hue_vals)
# %%
# # RANSAC
# # Initialize RANSAC regressor
# from sklearn.linear_model import RANSACRegressor
# ransac = RANSACRegressor(max_trials=10000, stop_probability=0.999,residual_threshold=(img.width*img.height/250))

# # Fit the model
# x = np.array(list(data.values())).reshape(-1,1)
# y = np.array(list(data.keys())).reshape(-1,1)
# ransac.fit(x,y )

# # Predict
# inlier_mask = ransac.inlier_mask_
# outlier_mask = np.logical_not(inlier_mask)

# # Plot the results
# lw = 2
# line_x = np.arange(0, 255).reshape(-1, 1)
# line_y_ransac = ransac.predict(line_x)

# plt.scatter(x[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.', label='Inliers')
# plt.scatter(x[outlier_mask], y[outlier_mask], color='gold', marker='.', label='Outliers')
# plt.plot(line_x, line_y_ransac, color='cornflowerblue', linewidth=lw, label='RANSAC regressor')
# plt.legend(loc='lower right')
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.show()
# print("outliers:",x[outlier_mask])

# def get_extrema(counts_list, half_width=2):
#     extrema = []
#     last_mean = 0
#     rising = False
#     means = []

#     for i in range(len(counts_list)):
#         if (i - half_width) < 0:
#             kernel_vals = counts_list[i - half_width :] + counts_list[: i + half_width + 1]
#             current_mean = sum(kernel_vals) / (2 * half_width + 1)
#         elif i > (len(counts_list) - 1 - half_width):
#             kernel_vals = counts_list[i - half_width :] + counts_list[: half_width - (len(counts_list) - (i + 1))]
#             current_mean = sum(kernel_vals) / (2 * half_width + 1)
#         else:
#             kernel_vals = counts_list[i - half_width : i + half_width + 1]
#             current_mean = sum(kernel_vals) / (2 * half_width + 1)
#         if current_mean >= last_mean:
#             rising = True
#         else:
#             if rising:
#                 # print(rising)
#                 extrema.append(i - 1)
#             rising = False

#         last_mean = current_mean
#         means.append(current_mean)

#         # print(rising)

#     return extrema, means


# plt.plot(counts.to_list())

# # Set the range of x-axis
# plt.xlim(0, 100)
# # Set the range of y-axis
# plt.ylim(0, 20000)

# # display plot
# plt.show()

# %%
