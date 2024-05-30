import numpy as np
import tensorflow as tf
import os
import pickle
import argparse
import json
from tqdm import tqdm

from MajorityVoteBounds.NeurIPS2020.optimize import optimize


def get_y_test(use_case):
    if use_case == 'cifar10':
        num_classes = 10
        from keras.datasets import cifar10
        _, (_, y_test) = cifar10.load_data()
        y_test = y_test[:, 0]
    elif use_case == 'cifar100':
        num_classes = 100
        from keras.datasets import cifar100
        _, (_, y_test) = cifar100.load_data()
        y_test = y_test[:, 0]
    elif use_case == 'imdb':
        num_classes = 1
        max_features = 20000
        from keras.datasets import imdb
        _, (_, y_test) = imdb.load_data(num_words=max_features)
    elif use_case == 'retinopathy':
        dataset_path = '../Datasets/Diabetic_Retinopathy'
        num_classes = 1
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        y_test = ImageDataGenerator().flow_from_directory(f'{dataset_path}/test', shuffle=False,
                                                          class_mode='binary').classes
    else:
        raise ValueError('Unknown use case')

    return y_test, num_classes


def get_prediction(y_pred, y_test, indices, weights, num_classes):
    num_models = len(indices)
    # y_pred has format (test_samples, models, classes)
    subset_y_pred = y_pred[:, indices, :]
    # Find indices of models that have nan predictions
    nan_indices = np.where(np.isnan(subset_y_pred).any(axis=(0, 2)))[0]
    if len(nan_indices) > 0:
        num_models -= len(nan_indices)
        # Remove nan predictions
        subset_y_pred = np.delete(subset_y_pred, nan_indices, axis=1)
        print(f'Removed {len(nan_indices)} models with nan predictions')
        if weights is not None:
            weights = np.delete(weights, nan_indices)
    # Get mean prediction
    subset_y_pred_ensemble = np.average(subset_y_pred, axis=1, weights=weights)
    # Majority voting (mode of the predictions)
    if num_classes == 1:
        subset_y_pred = subset_y_pred[:, :, 0]  # Just to remove the last dimension
        subset_y_pred_ensemble = subset_y_pred_ensemble[:, 0]  # Just to remove the last dimension

        subset_y_pred_argmax_per_model = (subset_y_pred > 0.5).astype(int)
        subset_y_pred_softmax_average = (subset_y_pred_ensemble > 0.5).astype(int)
        ensemble_loss = tf.keras.losses.BinaryCrossentropy()(y_test, subset_y_pred_ensemble).numpy()
        best_single_model_loss = np.min([tf.keras.losses.BinaryCrossentropy()(
            y_test, subset_y_pred[:, i]).numpy() for i in range(num_models)])

        diversity = (np.apply_along_axis(lambda x: len(np.unique(x)), axis=1,
                                         arr=subset_y_pred_argmax_per_model)) / 2
        ensemble_auc = tf.keras.metrics.AUC()(y_test, subset_y_pred_softmax_average).numpy()
    else:
        subset_y_pred_argmax_per_model = np.argmax(subset_y_pred, axis=2)
        subset_y_pred_softmax_average = np.argmax(subset_y_pred_ensemble, axis=1)
        ensemble_loss = tf.keras.losses.CategoricalCrossentropy()(tf.one_hot(y_test, num_classes),
                                                                  subset_y_pred_ensemble).numpy()
        best_single_model_loss = np.min([tf.keras.losses.CategoricalCrossentropy()(
            tf.one_hot(y_test, num_classes), subset_y_pred[:, i]).numpy() for i in range(num_models)])

        # Diversity as number of classes in ensemble vote out of all classes
        diversity = (np.apply_along_axis(lambda x: len(np.unique(x)), axis=1,
                                         arr=subset_y_pred_argmax_per_model) - 1) / (num_classes - 1)
        # AUC for multi-class classification not used
        ensemble_auc = -1.0

    ensemble_diversity = np.mean(diversity)
    subset_y_pred_maj_vote = np.array(
        [np.argmax(np.bincount(subset_y_pred_argmax_per_model[i, :], weights=weights)) for i in
         range(subset_y_pred_argmax_per_model.shape[0])], dtype=int)
    ensemble_acc_maj_vote = np.mean(subset_y_pred_maj_vote == y_test)
    ensemble_acc_softmax_average = np.mean(subset_y_pred_softmax_average == y_test)

    # Calculate the best model accuracy (only with actual test set if parts of test set are used for bound optimizing)
    best_single_model_accuracy = np.max([np.mean(subset_y_pred_argmax_per_model[:, i] == y_test)
                                         for i in range(num_models)])
    return (ensemble_acc_maj_vote,
            ensemble_acc_softmax_average,
            ensemble_loss,
            ensemble_auc,
            ensemble_diversity,
            best_single_model_accuracy,
            best_single_model_loss)


def load_all_predictions(folder: str, max_ensemble_size: int, test_pred_file_name='test_predictions.pkl'):
    # Get all subdirectories
    subdirs = sorted([f.path for f in os.scandir(folder) if f.is_dir()])
    # Load the models
    predictions = []
    for subdir in subdirs:
        # Find prediction files
        pred_files = sorted([f.path for f in os.scandir(subdir) if f.name.endswith(test_pred_file_name)])
        if len(pred_files) == 0:
            print(f'No predictions found in {subdir}')
            continue

        for i, pred_file in enumerate(pred_files):
            print(pred_file)
            # Load the predictions
            with open(pred_file, 'rb') as f:
                y_pred = pickle.load(f)
                # if 1D array, reshape to 2D array
                if len(y_pred.shape) == 1:
                    y_pred = y_pred.reshape(-1, 1)
            predictions.append(y_pred)
            if len(predictions) == max_ensemble_size:
                break
        if len(predictions) == max_ensemble_size:
            break

    # Concatenate all predictions to single array in the dimensions (test_samples, models, classes)
    y_pred = np.array(predictions).transpose(1, 0, 2)

    return y_pred


def ensemble_evaluation(use_case: str,
                        folder: str,
                        num_ensemble_members: int,
                        checkpoints_per_model:int,
                        reps: int,
                        pac_bayes: bool = True,
                        tta: bool = False,
                        test_set_cv: bool = True,
                        start_size=2):

    max_ensemble_size = num_ensemble_members * checkpoints_per_model
    y_test, num_classes = get_y_test(use_case)

    # Load the predictions
    y_pred = load_all_predictions(folder, max_ensemble_size)

    # Special case ub: Does not have last batch
    if len(y_pred) < len(y_test):
        # Load y_test from folder
        with open(os.path.join(folder, 'ub_y_true.pkl'), 'rb') as f:
            y_test = pickle.load(f)
    if tta:
        y_pred_tta, _, _ = load_all_predictions(folder, max_ensemble_size, 'test_tta_predictions.pkl')

    # "Test set cross-validation": Repeat 2 times to decrease variance (swap)
    # Same data set for all methods
    if test_set_cv:
        # Random select 50% of test set
        half = np.random.choice(y_test.shape[0], int(y_test.shape[0] / 2), replace=False)
        rest = np.setdiff1d(np.arange(y_test.shape[0]), half)
        test_splits = [(half, rest), (rest, half)]
    else:
        # Take full test set
        test_splits = [(np.arange(y_test.shape[0]), np.arange(y_test.shape[0]))]

    results = {}
    categories = ['uniform_last_per_model']
    if tta:
        categories.append('uniform_tta_last_per_model')
    if checkpoints_per_model > 1:
        categories.append('uniform_all_per_model')
    if pac_bayes:
        categories.append('tnd_last_per_model')
        if checkpoints_per_model > 1:
            categories.append('tnd_all_per_model')

    # Track the best single model accuracy (can change if test set is subsampled)
    best_single_model_accuracy = 0.0
    best_single_model_loss = np.inf

    for category in categories:

        print('Category:', category)

        ensemble_accs_mean = []
        ensemble_accs_std = []
        ensemble_losses_mean = []
        ensemble_losses_std = []
        ensemble_diversities_mean = []
        ensemble_diversities_std = []
        ensemble_auc_mean = []
        ensemble_auc_std = []
        for ensemble_size in tqdm(range(start_size, num_ensemble_members+1)):
            ensemble_accs_maj_vote = []
            ensemble_accs_softmax_average = []
            ensemble_losses = []
            ensemble_diversities = []
            ensemble_aucs = []
            for _ in range(reps):

                # Choose randomly ensemble_size integers from 0 to num_ensemble_members
                indices = np.random.choice(num_ensemble_members, ensemble_size, replace=False)

                if 'last_per_model' in category:
                    indices = [i * checkpoints_per_model + checkpoints_per_model - 1 for i in indices]
                elif 'all_per_model' in category:
                    indices = [i * checkpoints_per_model + j for i in indices for j in range(checkpoints_per_model)]

                per_ensemble_accs_maj_vote = []
                per_ensemble_accs_softmax_average = []
                per_ensemble_losses = []
                per_ensemble_diversities = []
                per_ensemble_auc = []
                for i, (test_risk_indices, test_bound_indices) in enumerate(test_splits):

                    if 'tta' in category:
                        y_pred_ = y_pred_tta[test_risk_indices]
                    else:
                        y_pred_ = y_pred[test_risk_indices]
                    y_test_ = y_test[test_risk_indices]

                    weights = None
                    if 'tnd' in category:
                        if 'last_per_model' in category:
                            save = len(indices) == num_ensemble_members
                        elif 'all_per_model' in category:
                            save = len(indices) == max_ensemble_size
                        else:
                            save = False
                        rhos, pac_results = optimize(use_case,
                                                     len(indices),
                                                     f'TEST_SET_{i}_{category}',
                                                     'iRProp',
                                                     1,
                                                     'DUMMY',
                                                     folder,
                                                     save,
                                                     indices=indices,
                                                     test_risk_indices=test_risk_indices,
                                                     test_bound_indices=test_bound_indices)
                        # Get the weights
                        weights = rhos[1]

                    (ensemble_acc_maj_vote,
                     ensemble_acc_softmax_average,
                     ensemble_loss,
                     ensemble_auc,
                     ensemble_diversity,
                     single_model_accuracy,
                     single_model_loss) = get_prediction(y_pred_, y_test_, indices, weights, num_classes)

                    if single_model_accuracy > best_single_model_accuracy:
                        best_single_model_accuracy = single_model_accuracy
                    if single_model_loss < best_single_model_loss:
                        best_single_model_loss = single_model_loss

                    per_ensemble_accs_maj_vote.append(ensemble_acc_maj_vote)
                    per_ensemble_accs_softmax_average.append(ensemble_acc_softmax_average)
                    per_ensemble_losses.append(ensemble_loss)
                    per_ensemble_diversities.append(ensemble_diversity)
                    per_ensemble_auc.append(ensemble_auc)

                ensemble_accs_maj_vote.append(np.mean(per_ensemble_accs_maj_vote))
                ensemble_accs_softmax_average.append(np.mean(per_ensemble_accs_softmax_average))
                ensemble_losses.append(np.mean(per_ensemble_losses))
                ensemble_diversities.append(np.mean(per_ensemble_diversities))
                ensemble_aucs.append(np.mean(per_ensemble_auc))

            ensemble_accs_mean.append((ensemble_size, np.mean(ensemble_accs_maj_vote), np.mean(ensemble_accs_softmax_average)))
            ensemble_accs_std.append((ensemble_size, np.std(ensemble_accs_maj_vote), np.std(ensemble_accs_softmax_average)))
            print('Mean Accuracy Majority Vote:', np.round(np.mean(ensemble_accs_maj_vote), 3),
                  '(Softmax Average:', np.round(np.mean(ensemble_accs_softmax_average), 3),
                  ') with', ensemble_size, 'models. Diversity:', np.round(np.mean(ensemble_diversities), 3))
            ensemble_losses_mean.append((ensemble_size, np.mean(ensemble_losses)))
            ensemble_losses_std.append((ensemble_size, np.std(ensemble_losses)))
            ensemble_diversities_mean.append((ensemble_size, np.mean(ensemble_diversities)))
            ensemble_diversities_std.append((ensemble_size, np.std(ensemble_diversities)))
            ensemble_auc_mean.append((ensemble_size, np.mean(ensemble_aucs)))
            ensemble_auc_std.append((ensemble_size, np.std(ensemble_aucs)))
        
        results[category] = (ensemble_accs_mean,
                             ensemble_accs_std,
                             ensemble_losses_mean,
                             ensemble_losses_std,
                             ensemble_diversities_mean,
                             ensemble_diversities_std,
                             ensemble_auc_mean,
                             ensemble_auc_std)

    # Save results
    #with open(os.path.join(folder, 'ensemble_results.pkl'), 'wb') as f:
    #    pickle.dump({
    #        'best_single_model_accuracy': best_single_model_accuracy,
    #        'best_single_model_loss': best_single_model_loss,
    #        'results': results,
    #    }, f)


def main():
    # Configuration
    parser = argparse.ArgumentParser(description='Ensemble prediction')
    parser.add_argument('--path', type=str, default='results/cifar10/resnet20/original',
                        help='Folder with the models for one experiment')
    parser.add_argument('-m', '--num_ensemble_members', type=int, default=50,
                        help='Number of ensemble members')
    parser.add_argument('-cp', '--checkpoints_per_model', type=int, default=1,
                        help='Number of checkpoints per independent model')
    parser.add_argument('--reps', type=int, help='Number of repetitions', default=5)
    parser.add_argument('--start_size', type=int, help='Start size of ensemble', default=2)

    args = parser.parse_args()
    path = args.path
    num_ensemble_members = args.num_ensemble_members
    checkpoints_per_model = args.checkpoints_per_model
    reps = args.reps
    start_size = args.start_size

    # Find any config.json in path recursively
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('config.json'):
                with open(os.path.join(root, file), 'r') as f:
                    config = json.load(f)
                    if 'use_case' not in config:
                        if 'imdb' in path:
                            use_case = 'imdb'
                        else:
                            raise ValueError('No use case found in config')
                    else:
                        use_case = config['use_case']
                    break

    if 'epoch_budget' in path:
        for i in range(start_size, num_ensemble_members+1):
            print(f'Ensemble size: {i}')
            path_ = os.path.join(path, f'{i:02d}')
            models = 0
            # Get number of models (in case not all are present)
            for root, dirs, files in os.walk(path_):
                for file in files:
                    if file.endswith('scores.json'):
                        models += 1
            if models == 0:
                print(f'No models found in {path_}')
                continue
            ensemble_evaluation(folder=path_,
                                num_ensemble_members=models,
                                checkpoints_per_model=checkpoints_per_model,
                                use_case=use_case,
                                reps=reps,
                                start_size=models)
    else:
        ensemble_evaluation(folder=path,
                            num_ensemble_members=num_ensemble_members,
                            checkpoints_per_model=checkpoints_per_model,
                            use_case=use_case,
                            reps=reps,
                            start_size=start_size)


if __name__ == '__main__':
    main()
