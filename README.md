# FIB data cleaner tool

### Why?

Focused Ion Beam milling based tomography is a powerful technique producing huge datasets.
Widely available tools (not this) work quite satisfactory when dataset is homogenous and there is limited drift between the taken slice images.
But what if FIB-SEM geometric limitations and requirements of results cause pathological cases where automatic alignement is not working (i.e. surface not perfectly perpendicular to FIB; subdivided takes on acquision with differing brightness/contrast; A need to sense a very subtile composition difference: High e-beam Current causing drifts)?   

### Capabilities
This tool is intended to be an intermediate tool in FIB data handling workflow.
It uses `Hyperspy` to load and save data (input/output).
The stack of images should be constructed before with Hyperspy (or ImageJ, or some SEM OEM software gives prestacked TIFF, which can be loaded directly).
This tool provides posibilities to:
* do simple lossless pixel-shift (`nupy.roll`) based alignment of slices
* correct pathological "skewed"-charging effect with affine transformation
* affine transformation can be defined by 3x3 matrix, or using a 3 draggable pivot points
* can lock a single slice, make it semi-transparent and compare with other slices
* can shift single or more selectable amount of slices simultaniously.
* PolyLine ROI tool for fast course alignment in one direction
* shows the views in (x,y) (y,z) and (x,z) perpendicular plainar views of the dataset. All views are updated after any slice shift or transformation. 
* shift instrucitons and affine transformation matric'es can be saved and loaded and applied to the initial dataset from simple json file. This way the raw data can be left unmodified. Albeit saving the modified (shifted/cropped/transformed) dataset as a new file will reset shift and transformation instructions and any new shift and/or transformation definitions should be applied no more to the raw/initial dataset but to such new modified copy of the data.
* has built-in simple normalisation (normalizes intensities to the ROI selected at x , y, to all z)
* has included QtConsole, which allows to use other Hyperspy functions.

### Requirements
Currently this is not a stand-alone application, and requires functioning python (preferably virtual) environment (conda/mamba venv or anything similar).
* GUI is based on `PyQt5` and for fast Graphical interaction uses `pyqtgraph` library.
* Requires `hyperspy` to load and save datasets.
* currently uses `scipy.ndimage` for affine transformations
* uses `affine6p` library for calculation of transformation matrix from sets of 3 original and moved points.

When having a working hyperspy environmet it is easy to meet above requiriments by `conda install` or `pip install` the `pyqtgraph` library, and `pip install` `affine6p`.

### Future considerations:
* Hyperspy is heavy, loading and saving could use `RossetaSciIO`, however Hyperspy loading and saving of files currently is much more easier.
* scipy.ndimage affine_transform is increadibly slow and live interaction is painful to use. 
* `opencv` could provide faster affine transform, and it has some interesting and useful filters (i.e. Bilateral), which could come handy with dealin with noisy FIB data.
* pyqtgraph has some interesting 3D capabilities and it is worth to explore.
