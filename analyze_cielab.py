# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from scipy.ndimage import median_filter
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


def lab_to_rgb(L, a, b):
    # Create a 1x1 LAB array
    lab_color = np.array([L, a, b], dtype=np.uint8).reshape((1, 1, 3))

    # Convert LAB to BGR using cv2
    bgr_color = cv2.cvtColor(lab_color, cv2.COLOR_LAB2BGR)

    # Convert BGR to RGB
    rgb_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2RGB)

    # Normalize RGB values to the range [0, 1]
    rgb_color_normalized = rgb_color / 255.0

    return rgb_color_normalized[0, 0, :]


def plot_luminance_stacked(dir_path, out_dir):
    fig, ax = plt.subplots()

    for filename in os.listdir(dir_path):
        path = os.path.join(dir_path, filename)
        # Open image with vv2 (assuming vv2 is your desired method)
        image = cv2.imread(path)

        # Convert to Lab color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        luminance = lab_image[:, :, 0]

        # Calculate the saturation distribution
        # The [:256] creates an array [0, 1, 2, ..., 255], and we use
        # it to select the values from sat_distribution. This ensures
        # that if there are missing values in the original distribution,
        # they are filled with zeros.
        lum_distribution = np.bincount(luminance.flatten(), minlength=256)[:256]
        lum_distribution = np.log(lum_distribution)
        ax.plot(lum_distribution, alpha=0.6, lw=0.5, c="black")

    ax.legend()
    ax.set_title("Luminance Distribution")
    ax.set_xlabel("Luminance")
    ax.set_ylabel("Frequency")
    plt.savefig(f"{out_dir}/luminance_distribution.png", dpi=300)
    plt.close()


def plot_luminance(path, filename, out_dir):
    fig_curr, ax_curr = plt.subplots()
    # Open image with vv2 (assuming vv2 is your desired method)
    image = cv2.imread(path)

    # Convert to Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance = lab_image[:, :, 0]

    # Calculate the saturation distribution
    # The [:256] creates an array [0, 1, 2, ..., 255], and we use
    # it to select the values from sat_distribution. This ensures
    # that if there are missing values in the original distribution,
    # they are filled with zeros.
    lum_distribution = np.bincount(luminance.flatten(), minlength=256)[:256]
    lum_distribution = np.log(lum_distribution)

    ax.plot(lum_distribution, alpha=0.6, lw=0.5)
    ax_curr.legend()
    ax_curr.set_title("Luminance Distribution")
    ax_curr.set_xlabel("Luminance")
    ax_curr.set_ylabel("Frequency")
    ax_curr.plot(lum_distribution)
    plt.savefig(f"{out_dir}/{filename.split('.')[0]}.png", dpi=300)
    plt.close()


def analyze_lab_colorplane(path):
    image = cv2.imread(path)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Access the a* and b* channels
    L, a, b = cv2.split(lab)

    # # mask low luminance pixels
    # lower = np.array([200, 0, 0])
    # upper = np.array([255, 255, 255])
    # mask = cv2.inRange(lab, lower, upper)

    # # save mask with masked pixels transparent
    # mask_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    # mask_image[:, :, 3] = mask
    # cv2.imwrite(f"sketching/{path.split('/')[-1]}.png", mask_image)

    # # apply mask
    # a = a[mask.nonzero()]
    # b = b[mask.nonzero()]

    # Flatten the channels to 1D arrays
    a_flat = a.flatten()
    b_flat = b.flatten()

    ab = np.column_stack((a_flat, b_flat))
    # remove duplicates
    ab = np.unique(ab, axis=0)
    # Define a distance threshold for points to be considered neighbors
    distance_threshold = 2  # Adjust as needed

    # Calculate pairwise distances between points
    distances = cdist(ab, ab)
    # Set diagonal elements (self distances) to a value larger than the distance threshold
    np.fill_diagonal(distances, np.inf)
    # Create a mask to identify points with at least two neighbors within the threshold
    neighbor_mask = np.sum(distances <= distance_threshold, axis=1) >= 12
    # Filter the point cloud based on the mask
    # print(ab.size)
    ab_masked = ab[neighbor_mask]
    # print(ab_masked.size)

    hull = ConvexHull(ab_masked)
    return a_flat, b_flat, ab_masked, hull


def plot_lab_colorplane(a_flat, b_flat, ab_masked, hull, filename, out_dir):
    # Add small jitter to identical points
    jitter_amount = 1.5  # Adjust as needed
    a_jittered = a_flat + np.random.uniform(
        low=-jitter_amount, high=jitter_amount, size=a_flat.shape
    )
    b_jittered = b_flat + np.random.uniform(
        low=-jitter_amount, high=jitter_amount, size=b_flat.shape
    )

    # get RBG color from a* and b* with fixed L
    L = 138
    rgb_colors = [lab_to_rgb(L, a_flat[i], b_flat[i]) for i in range(len(a_flat))]

    plt.figure(figsize=(8, 8))
    plt.scatter(a_jittered, b_jittered, c=rgb_colors, s=1, lw=0, alpha=0.3)
    plt.xlim(40, 220)
    plt.ylim(40, 220)
    plt.xlabel("a*")
    plt.ylabel("b*")

    # plot hull
    for simplex in hull.simplices:
        plt.plot(ab_masked[simplex, 0], ab_masked[simplex, 1], "k-", lw=0.5)

    # write out hull area
    plt.text(
        plt.xlim()[1] - 3,
        plt.ylim()[0] + 3,
        f"hull area: {hull.area:.2f}",
        color="black",
        ha="right",
        va="bottom",
    )

    plt.title(filename)
    # plt.show()
    plt.savefig(
        f'{out_dir}/{os.path.basename(filename).split(".")[0]}_plot.png', dpi=300
    )
    plt.close()
    print(f"{os.path.basename(filename)} finished")


import time

start = time.process_time()
dir_path = "data/downsampled"


meatadata = {}
with open("results.json", "r") as json_file:
    results = json.load(json_file)
# plot_luminance_stacked(dir_path, "out/lab_luminance")

for filename in os.listdir(dir_path):
    path = os.path.join(dir_path, filename)
    # path = "data/downsampled/df_dk_0010001_1725.tif"
    a_flat, b_flat, ab_masked, hull = analyze_lab_colorplane(path)
    # plot_lab_colorplane(
    #     a_flat, b_flat, ab_masked, hull, path, "out/cielab_ab_plots"
    # )

    # plot_luminance(path, filename, "out/lab_luminance")
    results[filename]["lab_analysis"] = {}
    # hull_width = np.max(hull.points[hull.simplices, 0]) - np.min(
    #     hull.points[hull.simplices, 0]
    # )
    # hull_height = np.max(hull.points[hull.simplices, 1]) - np.min(
    #     hull.points[hull.simplices, 1]
    # )

    results[filename]["lab_analysis"]["hull_area"] = hull.area
    results[filename]["lab_analysis"]["color_coded"] = True if hull.area > 85 else False

with open("results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)
print(time.process_time() - start)

# %%
