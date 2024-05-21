# Bayesian vs. PAC-Bayesian Deep Neural Network Ensembles

## Abstract
Bayesian neural networks address epistemic uncertainty by learning a posterior distribution over  model parameters. Sampling and weighting networks according to this posterior yields an ensemble model referred to as Bayes ensemble. 
Ensembles of neural networks (deep ensembles) can profit from the cancellation of errors effect: Errors by ensemble members may average out and the deep ensemble  achieves better predictive performance than each individual network. 
We argue that neither the sampling nor the weighting in a Bayes ensemble are particularly well-suited for increasing generalization performance, as they do not support the cancellation of errors effect, which is evident in the limit from the Bernstein-von~Mises theorem for misspecified models.
In contrast, a weighted average of models where the weights are optimized by minimizing a PAC-Bayesian generalization bound can improve generalization performance. This requires that the optimization takes correlations between models into account, which can be achieved by minimizing the tandem loss at the cost that hold-out data for estimating error correlations need to be available.
The PAC-Bayesian weighting increases the robustness against correlated models and models with lower performance in an ensemble. This allows to safely add several models from the same learning process to an ensemble, instead of using early-stopping for selecting a single weight configuration.
Our study presents empirical results supporting these conceptual considerations on four different classification datasets. We show that state-of-the-art Bayes ensembles from the literature, despite being computationally demanding, do not improve over simple uniformly weighted deep ensembles and cannot match the performance of deep ensembles weighted by optimizing the tandem loss, which additionally come with non-vacuous generalization guarantees.


## Requirements

### Virtual Environment
All experiments (except for IMDB) and the evaluation were run with Keras 3.1.0 and Tensorflow 2.16.1 using NVIDIA Cuda 12.3.
To install the requirements, run the following command with anaconda installed:

```setup
conda env create -f TF_KERAS_3_GPU_env.yml
conda activate TF_KERAS_3_GPU
```

For the IMDB dataset, the experiments were run with Keras 2.4.3 and Tensorflow 2.4.1 using NVIDIA Cuda 10.1.
To install the requirements, run the following command with anaconda installed:

```setup
conda env create -f TF_KERAS_GPU_env.yml
conda activate TF_KERAS_GPU
```

### Data Preparation

The datasets CIFAR-10, CIFAR-100 and IMDB are included in ```keras.datasets``` and will be downloaded automatically. 
For the EyePACS dataset:

1. Download the dataset from [Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) and extract the 
files. Download the labels for the validation and test set from 
[here](https://storage.googleapis.com/kaggle-forum-message-attachments/90528/2877/retinopathy_solution.csv).
2. The directory structure should look like this:
```retinopathy_solution.csv  sample/  sampleSubmission.csv  test_raw/  train_raw/  trainLabels.csv```
3. Run the following command to preprocess the dataset as decribed in the paper:
    
    ```data
    python .\util\load_data.py <path_to_data> --train --test
    ```
   This preprocesses the dataset and stores all samples in the respective train and test directories, with labels 
    as subdirectories. The size of the dataset is reduced drastically by this preprocessing step, from 89GB to 3GB.

## Training

To train the deep ensembles in the paper, run this command with the respective arguments:

```train
python -m training \
    --id="01" \
    --seed=1 \
    --out_folder="results/cifar10/resnet20/original" \
    --validation_split=0.0 \
    --model_type="ResNet20v1" \
    --data_augmentation \
    --nesterov \
    --optimizer="sgd" \
    --use_case="cifar10" \
    --initial_lr=0.1 \
    --l2_reg=0.002 \
    --lr_schedule="cifar" \
    --checkpointing \
    --checkpoint_every=40 \
    --epochs=200
```

For all hyperparameters and their respective values, please refer to the paper. Exemplary training scripts are 
given in the ```scripts``` directory (originally run on a SLURM cluster). 
```
scripts/
├── run_cifar.sh
├── run_epoch_budget.sh
├── run_retinopathy.sh
imdb/
├── run_imdb.sh
├── run_epoch_budget.sh
```

## Evaluation

To evaluate all models, run (either as script or single commands from the script):

```eval
./scripts/run_all_evaluations.bat
```
Then, to produce plots and tables, run:

```eval
./scripts/run_all_plots.bat
./scripts/run_all_tables.bat
```
