# coastsat

Python code to download publicly available satellite imagery with Google Earth Engine API and extract shorelines using a robust sub-pixel resolution shoreline detection algorithm described in *Vos K., Harley M.D., Splinter K.D., Simmons J.A., Turner I.L. (in review). Capturing intra-annual to multi-decadal shoreline variability from publicly available satellite imagery, Coastal Engineering*.

Written by *Kilian Vos*.

## Description

Satellite remote sensing can provide low-cost long-term shoreline data capable of resolving the temporal scales of interest to coastal scientists and engineers at sites where no in-situ measurements are available. Satellite imagery spannig the last 30 years with constant revisit periods is publicly available and suitable to extract repeated measurements of the shoreline positon.
*coastsat* is an open-source Python module that allows to extract shorelines from Landsat 5, Landsat 7, Landsat 8 and Sentinel-2 images.
The shoreline detection algorithm proposed here combines a sub-pixel border segmentation and an image classification component, which refines the segmentation into four distinct categories such that the shoreline detection is specific to the sand/water interface.

## Use 

A demonstration of the use of *coastsat* is provided in the Jupyter Notebook *shoreline_extraction.ipynb*. The code can also be run in Spyder with *main_spyder.py*.

The Python packages required to run this notebook can be installed by running the following anaconda command: *conda env create -f environment.yml*. This will create a new enviroment with all the relevant packages installed. You will also need to sign up for Google Earth Engine (https://earthengine.google.com and go to signup) and authenticate on the computer so that python can access via your login.

The first step is to retrieve the satellite images of the region of interest from Google Earth Engine servers by calling *SDS_download.get_images(sitename, polygon, dates, sat_list)*:
- *sitename* is a string which will define the name of the folder where the files will be stored
- *polygon* contains the coordinates of the region of interest (longitude/latitude pairs)
- *dates* defines the dates over which the images will be retrieved (e.g., *dates = ['2017-12-01', '2018-01-01']*)  
- *sat_list* indicates which satellite missions to consider (e.g., *sat_list = ['L5', 'L7', 'L8', 'S2']* will download images from Landsat 5, 7, 8 and Sentinel-2 collections).

The images are cropped on the Google Earth Engine servers and only the region of interest is downloaded resulting in low memory allocation (~ 1 megabyte/image for a 5km-long beach). The relevant image metadata (time of acquisition, geometric accuracy...etc) is stored in a file named *sitename_metadata.pkl*.

Once the images have been downloaded, the shorelines are extracted from the multispectral images using the sub-pixel resolution technique described in *Vos K., Harley M.D., Splinter K.D., Simmons J.A., Turner I.L. (in review). Capturing intra-annual to multi-decadal shoreline variability from publicly available satellite imagery, Coastal Engineering*.
The shoreline extraction is performed by the function SDS_shoreline.extract_shorelines(metadata, settings). The user must define the settings in a Python dictionary. To ensure maximum robustness of the algorithm the user can optionally digitize a reference shoreline (byc calling *SDS_preprocess.get_reference_sl(metadata, settings)*) that will then be used to identify obvious outliers and minimize false detections. Since the cloud mask is not perfect (especially in Sentinel-2 images) the user has the option to manually validate each detection by setting the *'check_detection'* parameter to *True*.
The shoreline coordinates (in the coordinate system defined by the user in *'output_epsg'* are stored in a file named *sitename_out.pkl*.

## Issues and Contributions

Looking to contribute to the code? Please see the [Issues page](https://github.com/kvos/coastsat/issues).
