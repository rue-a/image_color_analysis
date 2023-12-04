# %%

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
import pandas as pd
import cv2

image = cv2.imread("/home/service/sda3/alte_karten/data/original/MF_KuPL_380_0001.tif")


hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hues, saturations, values = cv2.split(hsv)


# make hue distribution
hue_distribution = np.bincount(hues.flatten(), minlength=180)[:180]


# Plot the histogram
plt.hist(hues.flatten(), bins=180, color="gray", edgecolor="none")
# plt.plot(hue_distribution, color='tomato')
# plt.ylim(0, 1500000)
plt.xlabel("Hue")
plt.ylabel("Frequency")
# %%
