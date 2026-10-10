Prozi - A Prosthetic Sizing Tool

- A computer vision-based tool for analyzing residual limb measurements and comparing 3D-printed prosthetic designs.

Overview:

Prozi is an independently developed Python application that explores how computer vision can assist with the initial sizing of upper-limb prosthetics. The software analyzes photographs of a residual limb, extracts physical measurements, and compares them with existing 3D-printed prosthetic models to help identify potential designs and estimate preliminary print scales.

The project began after I built and assembled an e-NABLE Raptor Reloaded prosthetic hand. Through that experience, I became interested in the challenges of determining the correct size for an individual recipient. I wanted to explore whether image processing could make parts of this process more accessible and efficient while still allowing experienced makers to review the results.

Watch a demonstration of Prozi's image-processing, measurement, and 3D model comparison workflow:
- https://youtu.be/bcev-vbtrv0

The demonstration uses AI-generated synthetic images. Prozi is an experimental prototype, and its sizing estimates have not been clinically validated.

How It Works: Prozi combines several stages of image processing and geometric analysis into one workflow.

1. Image Segmentation: Identifies the residual limb in side and top-view photographs, with the option to manually correct the selected area.

2. Ruler Calibration: Uses known ruler markings to convert pixel measurements into millimeters.

3. Limb Measurement: Calculates approximate length, width, thickness, taper, and changes in limb shape.

4. 3D Model Analysis: Processes STL files from open-source prosthetic designs and extracts geometric measurements from relevant components.

5. Model Comparison: Compares recipient measurements with prosthetic model geometry to identify potential candidates and calculate experimental uniform scaling estimates.

- The application is built using Python, OpenCV, NumPy, and STL geometry-processing tools. Its individual modules handle image processing, calibration, measurement, and model analysis.

Running the Project:

Install the core dependencies:
- pip install opencv-python numpy

Launch the application from the project directory: 
- python3 main.py

- The software guides users through photograph analysis and generates measurement and model comparison reports. Additional dependencies and model files may be required for STL analysis.

Current Development:

Prozi is a working proof of concept prototype. The image-processing and measurement workflow has been implemented, while the model-sizing algorithm is still being refined. One challenge identified during testing is that the external dimensions of a prosthetic model do not always represent the internal space available for fitting. Future improvements will focus on analyzing internal fitting geometry and validating scale estimates against known prosthetic builds. The current sizing estimates are experimental and should not be used to fabricate or fit a device without appropriate expert review.

Acknowledgments:

Prozi was independently developed and inspired by the work of the e-NABLE community, which provides access to open-source 3D-printed prosthetic designs. The Raptor Reloaded and other prosthetic designs referenced in this project belong to their respective creators. Prozi is not officially affiliated with or endorsed by e-NABLE. Original STL files are not included in this repository and should be obtained from their respective sources in accordance with their licenses.
