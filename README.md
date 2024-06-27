## Overview
The segmentation of lung tumors in medical imaging is a critical task in the field of medical diagnostics. This project leverages deep learning techniques, specifically the PyTorch framework, to achieve precise lung tumor segmentation.

![alt text](https://github.com/rbhardwaj2186/Lung_Tumor_Segmentation/blob/main/Images/images_2.gif)

## Preprocessing

1. Normalization: Given that CT images typically range from -1000 to 3071 HU, normalization is performed by dividing the pixel values by 3071, eliminating the need for mean and standard deviation computations.
2. Region of Interest (ROI) Extraction: To enhance focus on lung tumors, non-essential regions, such as parts of the lower abdomen, are excluded. For instance, the initial 30 slices (covering the lower abdomen to neck) can be omitted.
3. Dimensional Reduction: To reduce computational overhead, the task is approached on a slice level (2D) instead of a subject level (3D). Preprocessed data is stored as 2D slices, which speeds up the reading process compared to loading complete NIfTI files.
4. Resizing: Each slice and corresponding mask is resized to (256, 256) pixels. Nearest neighbor interpolation is applied when resizing masks to maintain label integrity.

## DataSet Creation
We need to implement the following functionality:
1. Create a list of all 2D slices. To so we need to extract all slices from all subjects
2. Extract the corresponding label path for each slice path
3. Load slice and label
4. Data Augmentation.
5. Return slice and mask <br/>
![alt text](https://github.com/rbhardwaj2186/Lung_Tumor_Segmentation/blob/main/Images/images_3.png)

## Model
then, we will create the model for the atrium segmentation! <br />
We will use the most famous architecture for this task, the U-NET (https://arxiv.org/abs/1505.04597). <br/>

The idea behind a UNET is the Encoder-Decoder architecture with additional skip-connctions on different levels:
The encoder reduces the size of the feature maps by using downconvolutional layers.
The decoder reconstructs a mask of the input shape over several layers by upsampling.
Additionally skip-connections allow a direct information flow from the encoder to the decoder on all intermediate levels of the UNET.
This allows for a high quality of the produced mask and simplifies the training process.<br />
![alt text](https://github.com/rbhardwaj2186/Lung_Tumor_Segmentation/blob/main/unet.png)

## Training
We will implement full segmentaion model with pytorch-lightning.
### Oversampling to tackle strong class imbalance
Lung tumors are often very small, thus we need to make sure that our model does not learn a trivial solution which simply outputs 0 for all voxels.<br />
We will use oversampling to sample slices which contain a tumor more often.

To do so we can use the **WeightedRandomSampler** provided by pytorch which needs a weight for each sample in the dataset.
### Loss

As this is a harder task to train you might try different loss functions:
We achieved best results by using the Binary Cross Entropy instead of the Dice Loss. <br/>
Computed Dice-score: 0.896

## Visualization

![alt text](https://github.com/rbhardwaj2186/Lung_Tumor_Segmentation/blob/main/Images/images_2.gif)
