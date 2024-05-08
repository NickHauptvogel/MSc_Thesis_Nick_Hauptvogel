import numpy as np
import tensorflow as tf
from keras.datasets import imdb, cifar10, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import pickle
import argparse
import sys
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from MajorityVoteBounds.NeurIPS2020.optimize import optimize

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
          'tab:gray', 'tab:olive', 'tab:cyan']

linestyles = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), (0, (3, 1, 1, 1))]

display_categories = {
    'uniform_last_per_model': 'Uniform last',
    'uniform_all_per_model': 'Uniform all',
    'tnd_last_per_model': 'TND last',
    'tnd_all_per_model': 'TND all',
}


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


def get_use_case_data(use_case: str, model_type: str = None):
    ylim_loss = (0., 1.)

    baseline_acc = None
    baseline_loss = None
    baseline_name = None
    if use_case == 'cifar10':
        ylim = (0.9, 0.975)
        num_classes = 10
        _, (_, y_test) = cifar10.load_data()
        y_test = y_test[:, 0]
        if model_type == 'ResNet20v1':
            ylim = (0.85, 0.95)
            ylim_loss = (0.15, 1.)
            baseline_acc = [0.9363]
            baseline_loss = [0.217]
            baseline_name = ['cSGHMC-ap[27] (Wenzel)']
        elif model_type == 'ResNet110v1':
            baseline_acc = [0.9554, 0.9637, 0.9531]
            baseline_name = ['SWAG[10] (Ashukha)', 'DE[10] (Ashukha)', 'SGD (Ashukha)']
        elif model_type == 'WideResNet28-10':
            baseline_acc = [0.9666, 0.9699, 0.963]
            baseline_name = ['SWAG[10] (Ashukha)', 'DE[10] (Ashukha)', 'SGD (Ashukha)']
    elif use_case == 'cifar100':
        ylim = (0.75, 0.84)
        num_classes = 100
        _, (_, y_test) = cifar100.load_data()
        y_test = y_test[:, 0]
        if model_type == 'ResNet110v1':
            baseline_acc = [0.7808, 0.8241, 0.7734]
            baseline_name = ['cSGLD[10] (Ashukha)', 'DE[10] (Ashukha)', 'SGD (Ashukha)']
        elif model_type == 'WideResNet28-10':
            baseline_acc = [0.8279, 0.8383, 0.8069]
            baseline_name = ['SWAG[10] (Ashukha)', 'DE[10] (Ashukha)', 'SGD (Ashukha)']
    elif use_case == 'imdb':
        baseline_acc = [0.8703]
        baseline_loss = [0.3044]
        baseline_name = ['cSGHMC-ap[7] (Wenzel)']
        ylim = (0.83, 0.88)
        ylim_loss = (0.25, 1.0)
        num_classes = 1
        max_features = 20000
        _, (_, y_test) = imdb.load_data(num_words=max_features)

    elif use_case == 'retinopathy':
        dataset_path = '../Datasets/Diabetic_Retinopathy'
        baseline_acc = [0.886, 0.903, 0.909, 0.916]
        baseline_name = ['MAP (Band)', 'DE[3] (Band)', 'MC-Dr. (Band)', 'MC-Dr.[3] (Band)']
        ylim = (0.86, 0.92)
        num_classes = 1
        y_test = ImageDataGenerator().flow_from_directory(f'{dataset_path}/test', shuffle=False,
                                                          class_mode='binary').classes
    else:
        raise ValueError('Unknown use case')

    return {
        'ylim': ylim,
        'ylim_loss': ylim_loss,
        'baseline_acc': baseline_acc,
        'baseline_loss': baseline_loss,
        'baseline_name': baseline_name,
        'num_classes': num_classes,
        'y_test': y_test
    }


def ensemble_prediction(use_case_data,
                        folder: str,
                        num_ensemble_members: int,
                        checkpoints_per_model:int,
                        use_case: str,
                        reps: int,
                        include_lam: bool,
                        pac_bayes: bool = True,
                        tta: bool = False,
                        test_set_cv: bool = True,
                        test_pred_file_name='test_predictions.pkl',
                        start_size=2,
                        overwrite_results: bool = False):

    if not overwrite_results and os.path.exists(os.path.join(folder, 'ensemble_results.pkl')):
        with open(os.path.join(folder, 'ensemble_results.pkl'), 'rb') as f:
            results = pickle.load(f)
            create_plots(use_case_data,
                         results['results'],
                         results['results'].keys(),
                         folder,
                         num_ensemble_members,
                         results['best_single_model_accuracy'],
                         results['best_single_model_loss'])
        return results['best_single_model_accuracy'], results['best_single_model_loss'], results['results']

    max_ensemble_size = num_ensemble_members * checkpoints_per_model
    y_test = use_case_data['y_test']

    # Load the predictions
    y_pred = load_all_predictions(folder, max_ensemble_size, test_pred_file_name)

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
        if include_lam:
            categories.append('lam_last_per_model')
        if checkpoints_per_model > 1:
            categories.append('tnd_all_per_model')
            if include_lam:
                categories.append('lam_all_per_model')

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
        for ensemble_size in tqdm(range(start_size, num_ensemble_members+1)):
            ensemble_accs_maj_vote = []
            ensemble_accs_softmax_average = []
            ensemble_losses = []
            ensemble_diversities = []
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
                for i, (test_risk_indices, test_bound_indices) in enumerate(test_splits):

                    if 'tta' in category:
                        y_pred_ = y_pred_tta[test_risk_indices]
                    else:
                        y_pred_ = y_pred[test_risk_indices]
                    y_test_ = y_test[test_risk_indices]

                    weights = None
                    if 'tnd' in category or 'lam' in category:
                        rhos, pac_results = optimize(use_case,
                                                     len(indices),
                                                     f'TEST_SET_{i}',
                                                     'iRProp',
                                                     1,
                                                     'DUMMY',
                                                     folder,
                                                     (len(indices) == max_ensemble_size),
                                                     indices=indices,
                                                     test_risk_indices=test_risk_indices,
                                                     test_bound_indices=test_bound_indices,
                                                     test_pred_file_name=test_pred_file_name)
                        # Get the weights
                        if 'tnd' in category:
                            weights = rhos[1]
                        elif 'lam' in category:
                            weights = rhos[0]

                    (ensemble_acc_maj_vote,
                     ensemble_acc_softmax_average,
                     ensemble_loss,
                     ensemble_diversity,
                     single_model_accuracy,
                     single_model_loss) = get_prediction(y_pred_, y_test_, indices, weights, use_case_data['num_classes'])

                    if single_model_accuracy > best_single_model_accuracy:
                        best_single_model_accuracy = single_model_accuracy
                    if single_model_loss < best_single_model_loss:
                        best_single_model_loss = single_model_loss

                    per_ensemble_accs_maj_vote.append(ensemble_acc_maj_vote)
                    per_ensemble_accs_softmax_average.append(ensemble_acc_softmax_average)
                    per_ensemble_losses.append(ensemble_loss)
                    per_ensemble_diversities.append(ensemble_diversity)
    
                ensemble_accs_maj_vote.append(np.mean(per_ensemble_accs_maj_vote))
                ensemble_accs_softmax_average.append(np.mean(per_ensemble_accs_softmax_average))
                ensemble_losses.append(np.mean(per_ensemble_losses))
                ensemble_diversities.append(np.mean(per_ensemble_diversities))

            ensemble_accs_mean.append((ensemble_size, np.mean(ensemble_accs_maj_vote), np.mean(ensemble_accs_softmax_average)))
            ensemble_accs_std.append((ensemble_size, np.std(ensemble_accs_maj_vote), np.std(ensemble_accs_softmax_average)))
            print('Mean Accuracy Majority Vote:', np.round(np.mean(ensemble_accs_maj_vote), 3),
                  '(Softmax Average:', np.round(np.mean(ensemble_accs_softmax_average), 3),
                  ') with', ensemble_size, 'models. Diversity:', np.round(np.mean(ensemble_diversities), 3))
            ensemble_losses_mean.append((ensemble_size, np.mean(ensemble_losses)))
            ensemble_losses_std.append((ensemble_size, np.std(ensemble_losses)))
            ensemble_diversities_mean.append((ensemble_size, np.mean(ensemble_diversities)))
            ensemble_diversities_std.append((ensemble_size, np.std(ensemble_diversities)))
        
        results[category] = (ensemble_accs_mean,
                             ensemble_accs_std,
                             ensemble_losses_mean,
                             ensemble_losses_std,
                             ensemble_diversities_mean,
                             ensemble_diversities_std)

    # Save results
    with open(os.path.join(folder, 'ensemble_results.pkl'), 'wb') as f:
        pickle.dump({
            'best_single_model_accuracy': best_single_model_accuracy,
            'best_single_model_loss': best_single_model_loss,
            'results': results,
        }, f)

    create_plots(use_case_data,
                 results,
                 categories,
                 folder,
                 num_ensemble_members,
                 best_single_model_accuracy,
                 best_single_model_loss)

    return best_single_model_accuracy, best_single_model_loss, results


def create_plots(use_case_data,
                 results,
                 categories,
                 folder,
                 num_ensemble_members,
                 best_single_model_accuracy=None,
                 best_single_model_loss=None):
    if num_ensemble_members > 50:
        every_ = 5
    elif num_ensemble_members > 30:
        every_ = 2
    else:
        every_ = 1

    figsize = (9, 6)

    # Plot the results
    for plot_majority_vote in [True, False]:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plt.setp(ax.spines.values(), color='#DDDDDD')
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8, zorder=0)
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
        ax.minorticks_on()

        min_mean = best_single_model_accuracy
        color_counter = 0
        for category in categories:
            ensemble_accs_mean, ensemble_accs_std, _, _, _, _ = results[category]
            ensemble_accs_mean = np.array(ensemble_accs_mean)
            ensemble_accs_std = np.array(ensemble_accs_std)

            if min_mean is None or min_mean > np.min(ensemble_accs_mean[:, 2]):
                min_mean = np.min(ensemble_accs_mean[:, 2])

            # Get a color
            color = colors[color_counter]
            color_counter = (color_counter + 1) % len(colors)
            if plot_majority_vote:
                plt.plot(ensemble_accs_mean[:, 0], ensemble_accs_mean[:, 2], label=f'{display_categories[category]}',color=color)
                plt.plot(ensemble_accs_mean[:, 0], ensemble_accs_mean[:, 1], color=color, linestyle='--')
                # Std as area around the mean
                plt.fill_between(ensemble_accs_mean[:, 0], ensemble_accs_mean[:, 1] - ensemble_accs_std[:, 1],
                                 ensemble_accs_mean[:, 1] + ensemble_accs_std[:, 1], alpha=0.3, color=color, zorder=3)
                plt.fill_between(ensemble_accs_mean[:, 0], ensemble_accs_mean[:, 2] - ensemble_accs_std[:, 2],
                                 ensemble_accs_mean[:, 2] + ensemble_accs_std[:, 2], alpha=0.3, color=color, zorder=3)
            else:
                plt.plot(ensemble_accs_mean[:, 0], ensemble_accs_mean[:, 2],
                         label=f'{display_categories[category]}', color=color)
                # Std as area around the mean
                plt.fill_between(ensemble_accs_mean[:, 0], ensemble_accs_mean[:, 2] - ensemble_accs_std[:, 2],
                                 ensemble_accs_mean[:, 2] + ensemble_accs_std[:, 2], alpha=0.3, color=color, zorder=3)
        plt.xlabel('Ensemble size')
        plt.ylabel('Accuracy')
        ylim = (max(min_mean - 0.005, use_case_data['ylim'][0]), use_case_data['ylim'][1])
        plt.ylim(ylim)
        # Horizontal line for the accuracy of the best model
        if best_single_model_accuracy is not None:
            plt.axhline(best_single_model_accuracy, color='orange', linestyle='--', label='Best individual model')
        if use_case_data['baseline_acc'] is not None:
            # Horizontal line for baseline accuracy
            for i, (acc, name) in enumerate(zip(use_case_data['baseline_acc'], use_case_data['baseline_name'])):
                # Get a linestyle
                linestyle = linestyles[(i % len(linestyles))]
                # Get a color
                greyscale = int(255 * i/len(use_case_data['baseline_acc']))
                color = f'#{greyscale:02x}{greyscale:02x}{greyscale:02x}'
                plt.axhline(acc, label=name, color=color, linestyle=linestyle)
        plt.title('Mean ensemble accuracy')
        plt.xticks(np.arange(ensemble_accs_mean[0][0], ensemble_accs_mean[-1][0] + 1, every_))
        plt.xlim(2, num_ensemble_members)
        ax.tick_params(color='#DDDDDD', which='both')
        plt.legend(loc='upper left')
        if plot_majority_vote:
            filename = 'ensemble_accs_majority_vote.pdf'
        else:
            filename = 'ensemble_accs.pdf'
        plt.tight_layout()
        plt.savefig(os.path.join(folder, filename))
        plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.setp(ax.spines.values(), color='#DDDDDD')
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8, zorder=0)
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    ax.minorticks_on()

    loss_max = best_single_model_loss
    for category in categories:
        _, _, ensemble_losses_mean, ensemble_losses_std, _, _ = results[category]
        if loss_max is None or loss_max < np.max(np.array(ensemble_losses_mean)[:, 1]):
            loss_max = np.max(np.array(ensemble_losses_mean)[:, 1])

        plt.plot(*zip(*ensemble_losses_mean), label=f'{display_categories[category]}')
        # Std as area around the mean
        plt.fill_between(np.array(ensemble_losses_mean)[:, 0],
                         np.array(ensemble_losses_mean)[:, 1] - np.array(ensemble_losses_std)[:, 1],
                         np.array(ensemble_losses_mean)[:, 1] + np.array(ensemble_losses_std)[:, 1], alpha=0.3, zorder=3)
    # Horizontal line for the loss of the best model
    if best_single_model_loss is not None:
        plt.axhline(best_single_model_loss, color='orange', linestyle='--', label='Best individual model')
    if use_case_data['baseline_loss'] is not None:
        # Horizontal line for baseline loss
        for i, (loss, name) in enumerate(zip(use_case_data['baseline_loss'], use_case_data['baseline_name'])):
            # Get a color
            greyscale = int(255 * i / len(use_case_data['baseline_acc']))
            color = f'#{greyscale:02x}{greyscale:02x}{greyscale:02x}'
            plt.axhline(loss, linestyle='--', label=name, color=color)
    plt.xlabel('Ensemble size')
    plt.ylabel('Categorical cross-entropy')
    if np.isfinite(loss_max):
        ylim_loss = (use_case_data['ylim_loss'][0], min(loss_max + 0.05, use_case_data['ylim_loss'][1]))
    else:
        ylim_loss = (use_case_data['ylim_loss'][0], use_case_data['ylim_loss'][1])
    plt.ylim(ylim_loss)
    plt.title('Mean ensemble loss')
    plt.xticks(np.arange(ensemble_losses_mean[0][0], ensemble_losses_mean[-1][0] + 1, every_))
    plt.xlim(2, num_ensemble_members)
    ax.tick_params(color='#DDDDDD', which='both')
    # Legend upper right
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'ensemble_losses.pdf'))
    plt.close()

    # Plot the results
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.setp(ax.spines.values(), color='#DDDDDD')
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8, zorder=0)
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    ax.minorticks_on()

    for category in categories:
        _, _, _, _, ensemble_diversity_mean, ensemble_diversity_std = results[category]

        plt.plot(*zip(*ensemble_diversity_mean), label=f'{display_categories[category]}')
        # Std as area around the mean
        plt.fill_between(np.array(ensemble_diversity_mean)[:, 0],
                         np.array(ensemble_diversity_mean)[:, 1] - np.array(ensemble_diversity_std)[:, 1],
                         np.array(ensemble_diversity_mean)[:, 1] + np.array(ensemble_diversity_std)[:, 1], alpha=0.3, zorder=3)

    plt.xlabel('Ensemble size')
    plt.ylabel('Diversity (Fraction of unique wrong classes present in prediction)')
    plt.ylim((0.001, 1.))
    plt.yscale('log')
    plt.title('Mean ensemble diversity')
    plt.xticks(np.arange(ensemble_diversity_mean[0][0], ensemble_diversity_mean[-1][0] + 1, every_))
    plt.xlim(2, num_ensemble_members)
    ax.tick_params(color='#DDDDDD', which='both')
    # Legend upper right
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'ensemble_diversities.pdf'))
    plt.close()


def main():
    # Configuration
    parser = argparse.ArgumentParser(description='Ensemble prediction')
    parser.add_argument('--folder', type=str, default='results/10_checkp_every_40_wenzel_0_2_val',
                        help='Folder with the models')
    parser.add_argument('--num_ensemble_members', type=int, default=50,
                        help='Number of ensemble members')
    parser.add_argument('--checkpoints_per_model', type=int, default=1, help='Number of checkpoints per independent model')
    parser.add_argument('--reps', type=int, help='Number of repetitions', required=False, default=5)
    parser.add_argument('--include_lam', action='store_true', help='Include lambda in the ensemble')

    args = parser.parse_args()
    folder = args.folder
    num_ensemble_members = args.num_ensemble_members
    checkpoints_per_model = args.checkpoints_per_model
    reps = args.reps
    include_lam = args.include_lam

    #folder = 'results/cifar10/wideresnet2810/original'
    folder = r"C:\Users\NHaup\OneDrive\Dokumente\Master_Studium\Semester_4\Thesis\Results\imdb\sse"
    # 80% of total number of models
    num_ensemble_members = 8
    checkpoints_per_model = 5
    reps = 5
    test_pred_file_name = 'test_predictions.pkl'
    # True to always overwrite results, False: Load results from file if exists
    overwrite_results = True

    # folder = 'results/retinopathy/resnet50/10_checkp_every_15_512_fullbatch_smalllr'
    # num_ensemble_members = 10
    # checkpoints_per_model = 6
    # reps = 3
    # test_pred_file_name = 'test_predictions.pkl'
    #test_pred_file_name = 'ub_y_pred.pkl'

    # Find any config.json in folder recursively
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('config.json'):
                with open(os.path.join(root, file), 'r') as f:
                    config = json.load(f)
                    if 'use_case' not in config:
                        if 'imdb' in folder:
                            use_case = 'imdb'
                        else:
                            raise ValueError('No use case found in config')
                    else:
                        use_case = config['use_case']
                    try:
                        model_type = config['model_type']
                    except KeyError:
                        model_type = config['model']
                    break

    use_case_data = get_use_case_data(use_case, model_type)

    if 'epoch_budget' in folder:
        results_dict = {}
        for i in range(2, num_ensemble_members+1):
            print(f'Ensemble size: {i}')
            folder_ = os.path.join(folder, f'{i:02d}')
            models = 0
            # Get number of models (in case not all are present)
            for root, dirs, files in os.walk(folder_):
                for file in files:
                    if file.endswith('scores.json'):
                        models += 1
            _, _, res = ensemble_prediction(use_case_data=use_case_data,
                                      folder=folder_,
                                      num_ensemble_members=models,
                                      checkpoints_per_model=checkpoints_per_model,
                                      use_case=use_case,
                                      reps=reps,
                                      include_lam=include_lam,
                                      test_pred_file_name=test_pred_file_name,
                                      start_size=models,
                                      overwrite_results=overwrite_results)
            for category in res.keys():
                if category not in results_dict:
                    results_dict[category] = ([], [], [], [], [], [])

                # Take last value of each metric (only biggest ensemble computed per epoch budget step)
                (mean, std, l_mean, l_std, div_mean, div_std) = res[category]
                results_dict[category][0].append((i, mean[-1][1], mean[-1][2]))
                results_dict[category][1].append((i, std[-1][1], std[-1][2]))
                results_dict[category][2].append((i, l_mean[-1][1]))
                results_dict[category][3].append((i, l_std[-1][1]))
                results_dict[category][4].append((i, div_mean[-1][1]))
                results_dict[category][5].append((i, div_std[-1][1]))
        create_plots(use_case_data, results_dict, results_dict.keys(), folder, num_ensemble_members)
    else:
        ensemble_prediction(use_case_data=use_case_data,
                            folder=folder,
                            num_ensemble_members=num_ensemble_members,
                            checkpoints_per_model=checkpoints_per_model,
                            use_case=use_case,
                            reps=reps,
                            include_lam=include_lam,
                            test_pred_file_name=test_pred_file_name,
                            overwrite_results=overwrite_results)


if __name__ == '__main__':
    main()
