# Efficient 3D affinely equivariant CNNs with adaptive fusion of augmented spherical Fourier-Bessel bases
PyTorch implementation of the paper "[Efficient 3D affinely equivariant CNNs with adaptive fusion of augmented spherical Fourier-Bessel bases](https://arxiv.org/abs/2402.16825)".

## Installation
The installation instructions are generally the same as [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet).

However, to use our convolutional layers, you will need to adjust the path to sphere_bessel.npy in the conv3d_sfb.py file appropriately for your environment.

## Usage

The template command for equivariance test
```bash
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD --equivariance_test
```


## Acknowledgement

[https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

[https://github.com/MIC-DKFZ/MedNeXt](https://github.com/MIC-DKFZ/MedNeXt)

[https://github.com/ZichenMiao/RotDCF](https://github.com/ZichenMiao/RotDCF)
