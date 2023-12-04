This repository contains scripts and outputs related to analyzing an image's color information.


## Directory/File Description

Core files are the CLIs `get_color_hull.py` and `plot_color_hull.py`. The first calculates the convex hull around the points that result when mapping the pixels color information into the a*-b* plane in the CIELab color space. The second CLI plot these points and colorizes them according to their color.

__Examples:__

`python get_color_hull.py data/original/df_dk_0010001_1725.tif --scale-factor 0.1`

`python plot_color_hull.py /home/service/sda3/alte_karten/data/original/28_6404_4_fbg.jpg --scale-factor 0.2 --jitter 0.5 --point-size 2 --draw-hull false`

| __Directory/File__ | __Description__ |
|-|-|
| data | Data directory (_due to size not included in this repo_). Contains the original raster maps in the folder _original_ and downsampled versions of the maps in _downsampled_.
| out/hsv_hue_plots | Results of the analysis of the hue values for each image. The plots depict the number of pixels for each possible hue value in black.  The transparent rectangles indicate the color of the according hue value for medium saturation and brightness values. | 
| out/cielab_ab_plots | Results of the analysis of the CIELAB (Lab) values for each image. The plots show the position of each pixel of the according image in the a*-b*-plane of the Lab color space. The pixels are colorized according to their a* and b* values at a medium-high Luminance (L-value). Since many pixels land at the same position, they are jittered and given a transparency. During analysis, outliers in the a*-b* plane are removed. The black line indicates the convex hull around all pixels in the a*-b*-plane, excluding outliers. | 
| sketching | Folder contains unordered scripts and outputs used for testing purposes.  | 
| get_color_hull.py | CLI to get the convex hull of the pixels of the image in the Lab a*-b* plane. |
| plot_color_hull.py | CLI to plot the the pixels of the image in the Lab a*-b* plane. |
| analyze_hsv.py | Script to analyze the Hue values of all images in the HSV color space and plot the results. |
| analyze_cielab.py | Script to analyze the a* and b* values of all images in the Lab color space and plot the results. |
| result_analysis.py | Script to analyze the gathered results. |
| downsample.sh | ImageMagick command used to downsample the original images. |
| gt.json | Groud Truth that was used to evaluate the results. | 
| results.json | Collected results for each image. |
| presentation.pptx | Presentation slides. | 

## This repository was created in the context of Dataxplorers Hackathon

__Dataxplorers Hackathon:__ Hackathon on Collaborative Solutions in Earth System, Biodiversity and Microbiota Research

### Challange 2: Inspection of color information in scanned raster maps - Team HuEL|O153

> __Challenge Provider:__
> Anna Lisa Schwartz & Peter Valena,
> Generaldirektion Staatliche Archive Bayerns,
> Leibniz Institute of Ecological Urban and Regional Development (IÖR)
>
> __Programming language:__ Free choice, preferably Python
>
> __Max. number of Teams:__ 3
>
> __Challenge Description:__
> Data on historical land use are essential for Earth Sciences and biodiversity research. Scanned topographic or cadastral maps are a valuable source of information on past land use. These scans usually come from libraries or archives and with heterogeneity in quality, scan resolution, format and color representation. However, when confronted with a collection of scanned maps, it is useful to know whether they contain color information (i.e. water bodies in blue, isohypsis in orange, vegetation in green, etc.) or in summary just grey value information. Since visual inspection is time consuming, an automation method is sought.
>
> It is important to note that, even if it is an image with colored bands, the map information itself might be just represented as black white (imagine the scan of a black white map on old yellowish paper).
>
> __Goal:__
> The goal is to implement a tool that allows for the automatic color inspection of directories or collections of scanned maps and their further sorting by color information or grey value information. Ideally, the tool also gives a hint on which land use classes might be represented in a map and extracts information on color, grey value as well as ideally land use class in a documented XML-structure.
>
> __Data:__
> Scanned historical maps.
