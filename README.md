# Learning Energy-Based Prior Model with Diffusion-Amortized MCMC
<img src="toy_example.PNG" alt="teaser" width="50%" />

[[Paper]()] [[Code](https://github.com/yuPeiyu98/DiffAMC)]

The official code repository for NeurIPS 2023 paper "Learning Energy-Based Prior Model with Diffusion-Amortized MCMC".

## Installation

The implementation depends on the following commonly used packages, all of which can be installed via conda.

| Package                  | Version                          |
| -------------------------|----------------------------------|
| PyTorch       		   | 1.10.0                           |
| pytorch-fid              | 0.2.1                            |
| pytorch-fid-wrapper      | 0.0.4                            |
| numpy                    | 1.21.0                           |

Please refer to this [repo](https://github.com/vict0rsch/pytorch-fid-wrapper) if you're having trouble installing pytorch-fid-wrapper.

## Datasets and Pre-trained Weights

Pretrained models are available at: https://drive.google.com/drive/folders/18UT4u4vco5TaEJx3HqksXyKP5l_jovUU?usp=sharing.

## Training

### Image Reconstruction and Generation
```bash
# Under the root folder
CUDA_VISIBLE_DEVICES=<GPU_ID> python train_gen_recon.py --dataset <DATASET_ALIAS> --seed <RANDOM_SEED> --log_path <PATH_FOR_TRAINED_WEIGHTS_AND_VIS> --data_path <PATH_TO_DATASETS>
```

One may want to specify the `log_path` argument for saving the trained weights and visualization results. Available dataset aliases include `(svhn, cifar10, celeba64, celebaHQ)`. `data_path` indicates the dataset location. L48-107 of `train_gen_recon.py` provide more details about how to set-up the `data_path` argument. Please find other available arguments at L352-405 in the `train_gen_recon.py` file. 

### Anomaly Detection
```bash
# Under the root folder
CUDA_VISIBLE_DEVICES=<GPU_ID> python train_anomaly_det.py --seed <RANDOM_SEED> --label <HELDOUT_DIGIT> --log_path <PATH_FOR_TRAINED_WEIGHTS_AND_VIS> --data_path <PATH_TO_DATASETS>
```
The `label` argument indicates the held-out digit in the MNIST dataset used for anomaly detection. Available options include `(1, 4, 5, 7, 9)`. `data_path` indicates the dataset location. L58-62 of `train_anomaly_det.py` provide more details about how to set-up the `data_path` argument.


Running these training scripts will automatically create the folders for the trained weights and other intermediate results in the `log_path`.

## Evaluation
To evaluate the pre-trained weights, one may consider using the following scripts  

### Image Reconstruction and Generation
```bash
# Under the root folder
CUDA_VISIBLE_DEVICES=<GPU_ID> python eval_gen_recon.py --dataset svhn --resume_path <PATH_TO_TRAINED_WEIGHTS> --data_path <PATH_TO_DATASETS> --e_l_step_size 0.4 --g_llhd_sigma 0.1

CUDA_VISIBLE_DEVICES=<GPU_ID> python eval_gen_recon.py --dataset cifar10 --resume_path <PATH_TO_TRAINED_WEIGHTS> --data_path <PATH_TO_DATASETS> --e_l_step_size 1.6 --g_llhd_sigma 0.1

CUDA_VISIBLE_DEVICES=<GPU_ID> python eval_gen_recon.py --dataset celeba64 --resume_path <PATH_TO_TRAINED_WEIGHTS> --data_path <PATH_TO_DATASETS> --e_l_step_size 0.4 --g_llhd_sigma 0.1

CUDA_VISIBLE_DEVICES=<GPU_ID> python eval_gen_recon.py --dataset celebaHQ --resume_path <PATH_TO_TRAINED_WEIGHTS> --data_path <PATH_TO_DATASETS> --e_l_step_size 0.4 --g_llhd_sigma 1.0
```

### Anomaly Detection
```bash
# Under the root folder
CUDA_VISIBLE_DEVICES=<GPU_ID> python eval_anomaly_det.py --label 1 --resume_path <PATH_TO_TRAINED_WEIGHTS> --data_path <PATH_TO_DATASETS> --g_llhd_sigma .1

CUDA_VISIBLE_DEVICES=<GPU_ID> python eval_anomaly_det.py --label 4 --resume_path <PATH_TO_TRAINED_WEIGHTS> --data_path <PATH_TO_DATASETS> --g_llhd_sigma 1.

CUDA_VISIBLE_DEVICES=<GPU_ID> python eval_anomaly_det.py --label 5 --resume_path <PATH_TO_TRAINED_WEIGHTS> --data_path <PATH_TO_DATASETS> --g_llhd_sigma 1.

CUDA_VISIBLE_DEVICES=<GPU_ID> python eval_anomaly_det.py --label 7 --resume_path <PATH_TO_TRAINED_WEIGHTS> --data_path <PATH_TO_DATASETS> --g_llhd_sigma 1.

CUDA_VISIBLE_DEVICES=<GPU_ID> python eval_anomaly_det.py --label 9 --resume_path <PATH_TO_TRAINED_WEIGHTS> --data_path <PATH_TO_DATASETS> --g_llhd_sigma 1.
```

### StyleGAN Inversion
```bash
# Under the root folder
CUDA_VISIBLE_DEVICES=<GPU_ID> python eval_stylegan_inv.py --dataset <DATASET_ALIAS> --resume_path <PATH_TO_TRAINED_WEIGHTS> --data_path <PATH_TO_DATASETS> --pretrained_G_path <TO_SPECIFY> --pretrained_E_path <TO_SPECIFY> --pretrained_F_path <TO_SPECIFY>
```

For styleGAN inversion, the `pretrained_G_path` is the path to the pre-trained generator weights, and the `pretrained_E_path` is the path to the encoder weights. `pretrained_F_path` specifies the path to the vgg model for perceptual loss. Available dataset aliases include `(ffhq, lsun_tower)`.

## Toy Example

#### Run the Code
To train the model on the toy example, one can run the following command in the `toy_example` folder.

```
CUDA_VISIBLE_DEVICES=<DEVICE_ID> python toy_example.py --seed <RANDOM_SEED_TO_SPECIFY> 
```
Here `--seed` argument specifies the random seed, which basically decides the ground-truth posterior distribution. The script will automatically generate a `logs/toy/<TIMESTAMP>` folder in the `toy_example` folder, where `<TIMESTAMP>` indicates the time you started this training process.

### Important Tips about Training
For most random seeds, we observed that our learned sampler could achieve decent approximation of the ground-truth posterior distributions obtained by long-run langevin dynamics within 300-3000 training iterations. This would take from several minutes to an hour or so on a NVIDIA RTX A6000 GPU. The training process takes ~2GB GPU memory. It is possible that there are some extreme cases where longer training iterations are needed to produce decent results.
For some random seeds, the default 1000-step langevin dynamics for sampling ground-truth posterior distribution might not converge. One may consider using 2000 or more steps by modifying the `g_l_steps` argument in the `sample_langevin_post_z` function at L277 in the `toy_example/toy_example.py`. One possible sign is that the `g_loss (avg) Q` (reconstruction error obtained by learned posterior samples) is significantly lower than `g_loss (avg) L` (reconstruction error obtained by langevin dynamics samples).

## Citation
```
@inproceedings{yu2023learning,
  author = {Yu, Peiyu and Zhu, Yaxuan and Xie, Sirui and Ma, Xiaojian and Gao, Ruiqi and Zhu, Song-Chun and Wu, Ying Nian},
  title = {Learning Energy-Based Prior Model with Diffusion-Amortized MCMC},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2023}
}
```