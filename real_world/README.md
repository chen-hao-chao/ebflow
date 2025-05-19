# Experiments on Real-World Datasets
This folder contains the code implementation of the experiments presented in Sections 5.2, 5.3 and 5.4 of the paper [*Training Energy-Based Normalizing Flow withScore-Matching Objectives*](https://arxiv.org/abs/2305.15267).

<img src="../assets/imputation_stl_celeb.png" alt= “” width="450">

## Setup
Installing the `ebflow` package allows you to run experiments with the `ebflow` command. Conduct the following instruction at the root of this directory to initiate the installation:
```
pip install -e .
```

## Usage

### Training Process

Use commands with the following format to train a model:
```
ebflow --config {$(1)} --loss {$(2)} --restore_path {$(3)}
```
- (1) `config`: training configuration (format: `{dataset}_{architecture}`).
- (2) `loss`: objective function (i.e., `ml`, `sml`, `ssm`, `dsm`, `fdssm`).
- (3) `restore_path`: the path to a checkpoint form which the training process resumes.

#### Examples:
- **(Results in Table 2)** 
  - Train EBFlow on MNIST with the FC-based architecture and the SSM objective.
  ```
  ebflow --config 'mnist_fc' --loss 'ssm'
  ```
  - Train EBFlow on CIFAR-10 with the CNN-based architecture and the DSM objective.
  ```
  ebflow --config 'cifar_cnn' --loss 'dsm'
  ```

- **(Results in Table 4)** Train EBFlow on MNIST with the FC-based architecture and the SSM objective.
  ```
  ebflow --config 'mnist_fc' --loss 'ssm' --withoutMaP
  ```

- **(Results in Figure 4)**
  - Train EBFlow on MNIST with the Glow architecture and the SSM / DSM objective.
  ```
  ebflow --config 'mnist_glow' --loss 'ssm'
  ebflow --config 'mnist_glow' --loss 'dsm'
  ```
  - Generate samples using a pretrained Glow model.
  ```
  ebflow --config 'mnist_glow_inv' --restore_path 'results/mnist_glow_ssm/checkpoints/checkpoint_best.tar'
  ebflow --config 'mnist_glow_inv' --restore_path 'results/mnist_glow_dsm/checkpoints/checkpoint_best.tar'
  ```
  > The Glow model can be trained more efficiently through DSM without compromising the performance (in comparison to SSM).

- **(Results in Figure 5)**
  - Train EBFlow on CelebA / STL-10 with the FC-based architecture and the DSM objective.
    ```
    ebflow --config 'celeb_fc' --loss 'dsm'
    ebflow --config 'stl_fc' --loss 'dsm'
    ```
  - Perform inplainting using a pretrained FC-based model.
    ```
    ebflow --config 'celeb_fc_mcmc' --restore_path 'results/celeb_fc_dsm/checkpoints/checkpoint_best.tar' --loss 'dsm'
    ebflow --config 'stl_fc_mcmc' --restore_path 'results/stl_fc_dsm/checkpoints/checkpoint_best.tar' --loss 'dsm'
    ```
  > Use `--restore_path {path_to_checkpoint}` to resume the training from a prior checkpoint if the training accidentally terminates.
  
  > Download CelebA dataset through [\[link\]](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

### Evaluation
#### Pretrained Weights
|  Dataset |  Loss   | Checkpoint       |  Dataset |  Loss   | Checkpoint       | 
| -------- | ------- | ---------------- | -------- | ------- | ---------------- |
| MNIST    |  ML     | [FC](https://drive.google.com/file/d/1KPhLou3InM1e3-8oqUZjqegqNmf-U42z/view?usp=sharing) / [CNN](https://drive.google.com/file/d/1HhHigOr_UWteIJHNOPf7g-06SkGcMuMn/view?usp=sharing) | CIFAR-10 |  ML     | [FC](https://drive.google.com/file/d/1Po1wXpb921NhQgXCUc2WzHRaV6kHE6ha/view?usp=sharing) / [CNN](https://drive.google.com/file/d/1ycCMu8nayXQwXeEXa_27fKkVX7X4G5gs/view?usp=sharing) |
| MNIST    |  SML    | [FC](https://drive.google.com/file/d/1wTp8DFhA-4gjd2LGNU4kg4vpnMahJgw-/view?usp=sharing) / [CNN](https://drive.google.com/file/d/1CoEGdkJMMiJVBk8pVcakhP_C3d0xQIU_/view?usp=sharing) | CIFAR-10 |  SML    | [FC](https://drive.google.com/file/d/1cDJXBoG7B1zrVq9ofFxWb5IKF0n_bdkr/view?usp=sharing) / [CNN](https://drive.google.com/file/d/1nMDIkJgbq2SqwlS9Yi-dpKdHk45WO1Q8/view?usp=sharing) |
| MNIST    |  DSM    | [FC](https://drive.google.com/file/d/1OsKx6_B99eOl1mGppVEEh591mkRQt6dB/view?usp=sharing) / [CNN](https://drive.google.com/drive/folders/1LkEHlPqlfFs1gG4LosPwPHFWTH48RM2h?usp=sharing) | CIFAR-10 |  DSM    | [FC](https://drive.google.com/file/d/19lZNDGRWjX_h57G3ej4H29qt6Gkk3EnM/view?usp=sharing) / [CNN](https://drive.google.com/file/d/1pRTlleJyBLKsHfljXIMu2Qxdj7tR5f2I/view?usp=sharing) |
| MNIST    |  SSM    | [FC](https://drive.google.com/file/d/1InG1sZqFQ9omaL58aSXIOfl5wTGiZG4o/view?usp=sharing) / [CNN](https://drive.google.com/file/d/1KvhYWvATepdUFNS7Jle8jktFH9k4_LwV/view?usp=sharing) | CIFAR-10 |  SSM    | [FC](https://drive.google.com/file/d/1wN07woXex2H-sGS3PjKL8uKjolm8PiCg/view?usp=sharing) / [CNN](https://drive.google.com/file/d/1AHBJ1ehytxNlCrEV-YDW2TOLyyZjdBIc/view?usp=sharing) |
| MNIST    |  FDSSM  | [FC](https://drive.google.com/file/d/1ZLJqhSAV8ix2H0DeprOY744D8wSjw9yb/view?usp=sharing) / [CNN](https://drive.google.com/file/d/1RGqEUxNHMjFxRpk6YLUn-0ILkeSqJ0t-/view?usp=sharing) | CIFAR-10 |  FDSSM  | [FC](https://drive.google.com/file/d/1kA5HIzjNlkb9KvsqBiQMNIY-j0FnimbG/view?usp=sharing) / [CNN](https://drive.google.com/file/d/1ycCMu8nayXQwXeEXa_27fKkVX7X4G5gs/view?usp=sharing) |

- Download the files using `gdown`
```
pip install gdown
gdown <file_id>
```
> `<file_id>` can be obtained from the url, e.g., `https://drive.google.com/file/d/<file_id>/view?usp=drive_link`.

#### Evaluating NLL
  - Evaluate the NLL of EBFlow trained with the CNN-based architecture and DSM on CIFAR-10.
  ```
  ebflow --config 'cifar_cnn' --loss 'dsm' --eval_only --restore_path <path_to_pretrained_weight>
  ```

### Details about the Code Implementation
- Models are built using the `FlowSequential` module (i.e. `ebflow/layers/flowsequential.py`). Each `FlowSequential` module contains a number of flow layers (i.e., `FlowLayer` and `LinearFlowLayer`), and has the following functions: *forward*, *reverse*, *log_prob*, *sample*.
- All of the flow layers belong to either `FlowLayer` or `LinearFlowLayer`. Layers with **constant Jacobians** are categorized as `LinearFlowLayer`, while the others are categorized as `FlowLayer`.
- The `Experiment` class (i.e., `ebflow/train/experiment.py`) handles training, evaluation, sampling, and checkpoint saving.
