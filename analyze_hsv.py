# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.patches as patches
import os
import json


def map_hue_to_color_name(hue_value):
    # Define hue ranges and corresponding color names
    hue_ranges = [
        (0, 15),  # Red
        (15, 25),  # Orange
        (25, 35),  # Yellow
        (35, 85),  # Green
        (85, 95),  # Turquoise
        (95, 125),  # Blue
        (125, 170),  # Violet
        (170, 180),  # Red
    ]

    color_names = [
        "Red",
        "Orange",
        "Yellow",
        "Green",
        "Turquoise",
        "Blue",
        "Violet",
        "Red",
    ]

    # Map the hue value to a color name
    for i, (start, end) in enumerate(hue_ranges):
        if start <= hue_value < end:
            return color_names[i]

    # If the hue doesn't fall into any defined range
    return "Unknown"


def find_hue_peaks(image_path):
    """
    Convert an image to HSV and analyze the distribution of
    hue values. The analysis ignores dark pixels (V < 50).
    """
    # read img
    image = cv2.imread(image_path)

    # Get the shape of the image
    height, width, channels = image.shape
    image_size = height * width

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hues, saturations, values = cv2.split(hsv)
    value_distribution = np.bincount(values.flatten(), minlength=256)[:256]
    sat_distribution = np.bincount(saturations.flatten(), minlength=256)[:256]

    # create a mask (cv2-HSV ranges: 0<=h<=179, 0<=s<=255, 0<=v<=255)
    lower = np.array([0, 0, 0])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # save mask with masked pixels transparent
    # mask_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    # mask_image[:, :, 3] = mask
    # cv2.imwrite(f"sketching/{image_path.split('/')[-1]}.png", mask_image)

    # apply mask
    hues = hues.flatten()[mask.flatten().nonzero()]

    # make hue distribution
    hue_distribution = np.bincount(hues.flatten(), minlength=180)[:180]
    # smooth distribution curve
    smoothed_distribution = gaussian_filter1d(hue_distribution, 2)
    # calc peaks
    peaks, _ = find_peaks(smoothed_distribution)

    # remove found peaks below a certain threshold
    threshold = (1 / 250) * image_size
    data = {
        val: key
        for key, val in list(
            zip(peaks, [smoothed_distribution[peak] for peak in peaks])
        )
    }

    # keep only peaks above threshold
    peaks_above = np.array([y for x, y in data.items() if x > threshold])
    peaks_below = [peak for peak in peaks if peak not in peaks_above]

    return (
        peaks_above,
        peaks_below,
        hue_distribution,
        smoothed_distribution,
        threshold,
        image_size,
        value_distribution,
        sat_distribution,
    )


def hue_to_rgb(hue):
    """creates a rgb hex color  code from a hue value"""
    # Create an HSV color with the specified hue, maximum saturation, and maximum value
    hsv_color = np.array([[hue, 255, 255]], dtype=np.uint8)
    rgb_color = cv2.cvtColor(np.array([hsv_color]), cv2.COLOR_HSV2RGB)[0][0]
    hex_color = "#{:02X}{:02X}{:02X}".format(*rgb_color)
    return hex_color


def make_plots(
    dir,
    peaks_x,
    peaks_below_x,
    hue_distribution,
    smoothed_distribution,
    threshold,
    image_size,
    caption,
):
    # Extract x and y values of peaks

    peaks_y = smoothed_distribution[peaks_x]
    peaks_below_y = smoothed_distribution[peaks_below_x]

    # Set up subplots
    fig, axs = plt.subplots(1, 4, figsize=(30, 5))

    y_lims = [
        round((11 / 10) * max(hue_distribution)),
        round(image_size / 10),
        round(image_size / 100),
        round(image_size / 500),
    ]
    for i in range(4):
        axs[i].set_xlim(0, 180)
        axs[i].set_ylim(0, y_lims[i])
        for x in peaks_x:
            rect = patches.Rectangle(
                (x - 5, 0),
                10,
                axs[i].get_ylim()[1],
                linewidth=0,
                edgecolor="none",
                facecolor=hue_to_rgb(x),
                alpha=0.1,
            )
            axs[i].add_patch(rect)
        axs[i].plot(
            smoothed_distribution,
            label="smoothed hue counts",
            color="orange",
            linewidth=2,
        )
        axs[i].plot(hue_distribution, label="hue counts", color="black", linewidth=0.5)
        axs[i].plot(peaks_x, peaks_y, "x", color="red", label="Peaks")
        axs[i].plot(
            peaks_below_x,
            peaks_below_y,
            "x",
            color="coral",
            label="Peaks Below Threshold",
        )
        axs[i].axhline(
            y=threshold, color="gray", linestyle="--", label=f"Threshold ({threshold})"
        )
        axs[i].set_xlabel("hue")
        axs[i].set_ylabel("count")
        axs[i].legend()

    fig.suptitle(caption, fontsize=14)
    # plt.show()
    plt.savefig(f'{dir}/{os.path.basename(caption).split(".")[0]}_plot.png', dpi=300)
    plt.close()


results = {}
with open("results.json", "r") as json_file:
    results = json.load(json_file)
dir_path = "data/original"
plots_dir = "out/hsv_hue_plots"
for filename in os.listdir(dir_path):
    path = os.path.join(dir_path, filename)
    (
        peaks,
        peaks_below,
        hue_distribution,
        smoothed_distribution,
        threshold,
        image_size,
        value_distribution,
        sat_distribution,
    ) = find_hue_peaks(path)
    # print(len(peaks))

    results[filename]["hue_analysis"] = {}
    results[filename]["hue_analysis"]["color_coded"] = True if len(peaks) > 1 else False
    results[filename]["hue_analysis"]["found_peaks"] = [peak / 180 for peak in peaks]
    results[filename]["hue_analysis"]["found_peaks_colors"] = [
        map_hue_to_color_name(peak) for peak in peaks
    ]

    # make_plots(
    #     plots_dir,
    #     peaks,
    #     peaks_below,
    #     hue_distribution,
    #     smoothed_distribution,
    #     threshold,
    #     image_size,
    #     path,
    # )
    # plt.plot(value_distribution, c="seagreen")
    # plt.plot(sat_distribution, c="tomato")
    # break
with open("results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)  #
# plt.xlabel("saturation/value")
# plt.ylabel("frequency")
# plt.savefig(f"{plots_dir}/sat_val_freq.png")
# plt.close()
