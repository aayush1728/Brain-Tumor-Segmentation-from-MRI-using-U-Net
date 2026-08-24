# Brain MRI Tumour Segmentation Using U-Net

## Overview

This project implements a **U-Net-based image segmentation pipeline**
for detecting and segmenting brain tumours in MRI images.

The notebook uses the **LGG MRI Segmentation** dataset and follows a
complete workflow:

1.  Dataset inspection and exploratory data analysis
2.  MRI image and tumour-mask preparation
3.  Train/validation/test split
4.  Image augmentation and preprocessing
5.  U-Net model construction from scratch
6.  Model training with Dice loss
7.  Evaluation using IoU, Dice coefficient, and binary accuracy
8.  Model and architecture export
9.  Tumour-mask prediction on unseen test images

The original notebook was developed in a Kaggle environment and uses
TensorFlow/Keras for deep-learning implementation.

------------------------------------------------------------------------

## Objective

The objective is to perform **pixel-wise segmentation** of tumour
regions in brain MRI images.

Unlike image classification, where a complete image is assigned a label,
semantic segmentation predicts a class for every pixel. In this project,
the U-Net model predicts a binary mask corresponding to the tumour
region.

------------------------------------------------------------------------

## Dataset

The notebook uses the **LGG MRI Segmentation** dataset available through
Kaggle.

The dataset contains brain MRI images together with corresponding binary
tumour masks. The notebook also reads a `data.csv` file containing
patient-level metadata.

The data directory used in the original notebook is:

``` text
/kaggle/input/lgg-mri-segmentation/kaggle_3m/
```

The image files are stored as `.tif` files, with corresponding
segmentation masks identified using the `_mask.tif` suffix.

### Dataset inspection

The notebook initially loads:

``` python
img_data = pd.read_csv(
    '../input/lgg-mri-segmentation/kaggle_3m/data.csv'
)
```

The metadata table contains 110 patient records and 18 columns in the
executed notebook.

The image/mask inventory contains:

-   **3,929 original MRI images**
-   **3,929 corresponding masks**
-   **7,858 image/mask-related file entries**

The notebook creates a dataframe with:

``` text
patient_id
img_path
mask_path
mask
```

where `mask` indicates whether a tumour is present in the corresponding
segmentation mask.

------------------------------------------------------------------------

## Data Preparation

### Image and mask separation

The notebook separates original MRI images from segmentation masks using
the filename:

``` python
original_img = df[~df['img_path'].str.contains("mask")]
mask_img = df[df['img_path'].str.contains("mask")]
```

The images and masks are then sorted so that corresponding image-mask
pairs are aligned.

### Tumour presence detection

A binary tumour-presence indicator is generated from each mask:

``` python
def get_diagnosis(img_path):
    value = np.max(cv2.imread(img_path))
    if value > 0:
        return 1
    else:
        return 0
```

A value of:

-   `0` → no tumour mask
-   `1` → tumour mask present

### Class distribution

The executed notebook reports:

``` text
No tumour: 2556
Tumour:    1373
```

This gives a total of 3,929 image-mask pairs.

------------------------------------------------------------------------

## Train / Validation / Test Split

The dataset is divided using `train_test_split`.

The notebook first reserves 10% for testing:

``` python
mri_train, mri_test = train_test_split(
    mri_df,
    test_size=0.1
)
```

The remaining training data is further split into training and
validation subsets:

``` python
mri_train, mri_val = train_test_split(
    mri_train,
    test_size=0.2
)
```

The executed notebook reports:

``` text
Training data: 2828 samples
Test data:      393 samples
```

The validation set is created separately from the training set but its
printed size is not explicitly reported in the notebook output.

> **Note:** The split is performed without an explicit `random_state`,
> so the exact partition can change between executions.

------------------------------------------------------------------------

## Image Preprocessing and Data Augmentation

Images are resized to:

``` text
256 × 256 pixels
```

The training generator applies data augmentation using:

``` python
train_generator_args = dict(
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

The generator uses a batch size of:

``` text
32
```

The notebook normalizes image and mask values using:

``` python
def adjust_data(img, mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return img, mask
```

Thus, the target segmentation mask is converted into a binary
representation.

------------------------------------------------------------------------

# U-Net Architecture

## Why U-Net?

U-Net is a convolutional neural-network architecture designed for
biomedical image segmentation. It uses an encoder-decoder structure with
skip connections between corresponding encoder and decoder levels.

The architecture implemented in this notebook follows the characteristic
U-shaped design:

``` text
Input
  │
  ├── Encoder
  │     ├── Conv → Conv → Pool
  │     ├── Conv → Conv → Pool
  │     ├── Conv → Conv → Pool
  │     └── Conv → Conv → Pool
  │
  ├── Bottleneck
  │
  └── Decoder
        ├── Upsample → Concatenate → Conv → Conv
        ├── Upsample → Concatenate → Conv → Conv
        ├── Upsample → Concatenate → Conv → Conv
        └── Upsample → Concatenate → Conv → Conv
                    │
                    ▼
              Sigmoid Output
```

## Input

The model accepts RGB images with dimensions:

``` text
256 × 256 × 3
```

defined by:

``` python
input_size = (256, 256, 3)
```

## Encoder

The encoder progressively increases the number of convolutional filters:

``` text
64 → 128 → 256 → 512
```

Each encoder block uses convolutional layers followed by batch
normalization and ReLU activation, with max pooling used for spatial
downsampling.

## Bottleneck

The deepest part of the network uses:

``` text
1024 convolutional filters
```

This provides the highest-level feature representation before
reconstruction begins.

## Decoder

The decoder progressively restores the spatial resolution:

``` text
512 → 256 → 128 → 64
```

Skip connections concatenate encoder features with decoder features at
corresponding spatial scales.

## Output

The final layer is:

``` python
Conv2D(1, 1, activation='sigmoid')
```

Therefore, the network produces a single-channel probability mask:

``` text
256 × 256 × 1
```

where each pixel represents the estimated probability of belonging to
the tumour region.

------------------------------------------------------------------------

# Training

## Loss Function

The notebook uses the **Dice coefficient loss**:

``` python
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
```

The Dice coefficient is calculated as:

\[ Dice = `\frac{2|A \cap B| + s}`{=tex} {\|A\| + \|B\| + s} \]

where `s = 100` is the smoothing constant used in the notebook.

The Dice loss is therefore:

\[ L\_{Dice} = 1 - Dice \]

Dice-based losses are particularly useful for segmentation problems
because they directly measure overlap between predicted and reference
regions.

------------------------------------------------------------------------

## Evaluation Metrics

The notebook tracks:

### Binary Accuracy

``` python
"binary_accuracy"
```

### Intersection over Union

The implemented IoU metric is:

``` python
def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (
        sum_ - intersection + smooth
    )
    return jac
```

Mathematically:

\[ IoU = `\frac{|A \cap B| + s}`{=tex} {\|A `\cup `{=tex}B\| + s} \]

where `s = 100`.

### Dice Coefficient

The notebook also tracks:

``` python
dice_coef
```

------------------------------------------------------------------------

# Optimizer and Training Configuration

The model is trained using the Adam optimizer:

``` python
Adam(
    learning_rate=1e-3,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=None,
    decay=1e-3/32,
    amsgrad=False
)
```

The configured number of epochs is:

``` text
100
```

The batch size is:

``` text
32
```

The model is compiled using:

``` python
model.compile(
    optimizer=Adam(...),
    loss=dice_coef_loss,
    metrics=[
        "binary_accuracy",
        iou,
        dice_coef
    ]
)
```

------------------------------------------------------------------------

# Callbacks

The notebook configures callbacks for training, including learning-rate
reduction and model checkpointing.

A learning-rate reduction callback monitors validation IoU:

``` python
ReduceLROnPlateau(
    monitor='val_iou',
    patience=8,
    ...
)
```

This allows the learning rate to be reduced when validation IoU stops
improving.

------------------------------------------------------------------------

# Model Saving

The trained model is saved as:

``` text
saved_model/my_model.h5
```

The notebook also exports the model architecture to JSON:

``` text
UNet-seg-model.json
```

This separates the network architecture from the trained model weights
and allows the architecture to be reconstructed later.

------------------------------------------------------------------------

# Prediction

For prediction, each test MRI image is:

1.  Read from disk
2.  Resized to `256 × 256`
3.  Converted to a floating-point array
4.  Standardized using its mean and standard deviation
5.  Expanded to include the batch dimension
6.  Passed through the trained U-Net

The prediction function creates:

``` text
img_path
predicted_mask
has_mask
```

The model output is interpreted as a tumour mask after rounding the
predicted probabilities.

If the predicted mask contains no positive pixels:

``` text
has_mask = 0
```

otherwise:

``` text
has_mask = 1
```

------------------------------------------------------------------------

# End-to-End Workflow

``` text
                 LGG MRI Dataset
                        │
                        ▼
              Load MRI + Mask Files
                        │
                        ▼
             Pair Images and Masks
                        │
                        ▼
             Determine Tumour Presence
                        │
                        ▼
          Train / Validation / Test Split
                        │
                        ▼
             Image Preprocessing
                        │
                        ▼
              Data Augmentation
                        │
                        ▼
                 U-Net Encoder
                        │
                        ▼
                   Bottleneck
                        │
                        ▼
                 U-Net Decoder
                        │
                        ▼
             Predicted Tumour Mask
                        │
                        ▼
        IoU / Dice / Binary Accuracy
                        │
                        ▼
             Test-Image Prediction
```

------------------------------------------------------------------------

# Technologies Used

  Category                     Tools / Libraries
  ---------------------------- ----------------------
  Programming Language         Python
  Deep Learning                TensorFlow, Keras
  Numerical Computing          NumPy
  Data Processing              Pandas
  Image Processing             OpenCV, scikit-image
  Visualization                Matplotlib, Seaborn
  Machine Learning Utilities   scikit-learn
  Dataset Environment          Kaggle
  Model Architecture           U-Net

------------------------------------------------------------------------

# Project Structure

A suggested project structure based on the notebook is:

``` text
brain-mri-unet/
│
├── data/
│   └── kaggle_3m/
│
├── notebooks/
│   └── unet-from-scratch-segmentation-tumour.ipynb
│
├── models/
│   ├── my_model.h5
│   └── UNet-seg-model.json
│
├── outputs/
│   ├── predictions/
│   └── visualizations/
│
└── README.md
```

------------------------------------------------------------------------

# Key Results Reported in the Notebook

The notebook establishes the following dataset-level results during
execution:

  Quantity                               Value
  ---------------------------- ---------------
  Patient metadata records                 110
  Image-mask pairs                       3,929
  Images without tumour mask             2,556
  Images with tumour mask                1,373
  Training samples                       2,828
  Test samples                             393
  Image size used by model           256 × 256
  Batch size                                32
  Configured epochs                        100
  Initial learning rate                  0.001
  Model output                   256 × 256 × 1

The notebook plots the training history for IoU, Dice coefficient, loss,
and validation performance. Specific final metric values should be taken
from the actual executed training history rather than inferred from the
notebook configuration.

------------------------------------------------------------------------

# Important Implementation Notes

## 1. Dataset splitting

The notebook performs image-level random splitting rather than
explicitly grouping samples by patient.

Because multiple MRI slices can belong to the same patient, a
patient-level split may be preferable when evaluating generalization to
unseen patients.

## 2. Standardization inconsistency

The training generator normalizes images using division by 255, while
the prediction function additionally standardizes each test image using:

``` python
img -= img.mean()
img /= img.std()
```

For a production-quality pipeline, training and inference preprocessing
should be made consistent.

## 3. Random seed

The notebook does not specify a fixed `random_state` for the
train/validation/test split. Consequently, the exact partitions may
differ between runs.

## 4. Medical interpretation

The model is a research/educational segmentation implementation. Its
predictions should not be interpreted as a clinical diagnosis without
appropriate medical validation.

------------------------------------------------------------------------

# References

1.  **Ronneberger, O., Fischer, P., & Brox, T. (2015).**\
    *U-Net: Convolutional Networks for Biomedical Image Segmentation.*\
    Medical Image Computing and Computer-Assisted Intervention (MICCAI),
    234--241.\
    https://doi.org/10.1007/978-3-319-24574-4_28

2.  **Mayo Clinic.**\
    *Brain tumor -- Symptoms and causes.*\
    The original notebook cites Mayo Clinic for its introductory
    description of brain tumours and uses a Mayo Clinic image as the
    introductory visual.\
    https://www.mayoclinic.org/diseases-conditions/brain-tumor/symptoms-causes/syc-20350084

3.  **zhixuhao/unet.**\
    The original notebook provides the GitHub implementation as a
    reference for the U-Net model.\
    https://github.com/zhixuhao/unet

------------------------------------------------------------------------

## Source

This documentation was prepared from the uploaded notebook:

`unet-from-scratch-segmentation-tumour.ipynb`

The notebook records an implementation of brain MRI tumour segmentation
using a U-Net architecture, including data preparation, model
construction, training, evaluation, saving, and prediction.
