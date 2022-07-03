# Wake-Sleep-ABP

## Installation

The implementation depends on the following commonly used packages, all of which can be installed via conda.

| Package       | Version                          |
| ------------- | -------------------------------- |
| PyTorch       | ≥ 1.8.1                          |
| numpy         | 1.21.0                           |
| opencv-python | 4.5.1.48                         |
| pandas        | 1.2.3                            |

## Data 

To train the model on CIFAR-10 dataset, please download the file from https://drive.google.com/file/d/1PmKrfeCx_LHWcmhoqESosYGMmYw_MjKH/view?usp=sharing, and place the decompressed files under `data/CIFAR10` folder.

## Training

```bash
# Under the root folder
python main.py --checkpoints <TO_BE_SPECIFIED>
```

You may specify the value of arguments during training. Please find the available arguments in the `config.yml.example` file in the  `workspace` folder.

`DATA` indicates the dataset to use (`CIFAR10`, `MNIST`). The path to your dataset folder, i.e., `ROOT_DIR`, needs to be specified before running the script.

## To-do list

Implementing evaluation protocols for likelihood estimation and FID score computation.