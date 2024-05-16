import argparse
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
import numpy as np
import pandas as pd
import os
import re
import pickle


linestyles = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), (0, (3, 1, 1, 1))]

display_categories = {
    'uniform_last_per_model': r'AVG$_u$ last',
    'uniform_all_per_model': r'AVG$_u$ all',
    'tnd_last_per_model': r'AVG$_\rho$ last',
    'tnd_all_per_model': r'AVG$_\rho$ all',
}

colors = {
    'uniform_last_per_model': 'darkorange',
    'uniform_all_per_model': 'peru',
    'tnd_last_per_model': 'olivedrab',
    'tnd_all_per_model': 'darkolivegreen',
}

use_case_display = {
    'cifar10': 'CIFAR-10',
    'cifar100': 'CIFAR-100',
    'imdb': 'IMDB',
    'retinopathy': 'EyePACS'
}


def get_use_case_data(use_case: str, model_type: str = None):
    ylim_loss = (0., 1.)

    baseline_acc = None
    baseline_loss = None
    baseline_name = None
    if use_case == 'cifar10':
        ylim = (0.9, 0.975)
        if model_type == 'ResNet20v1':
            ylim = (0.85, 0.95)
            ylim_loss = (0.15, 1.)
            baseline_acc = [0.9363]
            baseline_loss = [0.217]
            baseline_name = ['cSGHMC-ap[27]']
        elif model_type == 'ResNet110v1':
            baseline_acc = [0.9554, 0.9637, 0.9531]
            baseline_name = ['SWAG[10]', 'DE[10]', 'SGD']
        elif model_type == 'WideResNet28-10':
            baseline_acc = [0.9666, 0.9699, 0.963]
            baseline_name = ['SWAG[10]', 'DE[10]', 'SGD']
    elif use_case == 'cifar100':
        ylim = (0.70, 0.84)
        if model_type == 'ResNet110v1':
            baseline_acc = [0.7808, 0.8241, 0.7734]
            baseline_name = ['cSGLD[10]', 'DE[10]', 'SGD']
        elif model_type == 'WideResNet28-10':
            baseline_acc = [0.8279, 0.8383, 0.8069]
            baseline_name = ['SWAG[10]', 'DE[10]', 'SGD']
    elif use_case == 'imdb':
        baseline_acc = [0.8703]
        baseline_loss = [0.3044]
        baseline_name = ['cSGHMC-ap[7]']
        ylim = (0.83, 0.88)
        ylim_loss = (0.25, 1.0)
    elif use_case == 'retinopathy':
        baseline_acc = [0.909, 0.916, 0.886, 0.903]
        baseline_name = ['MC-Dr. [5]', 'MC-Dr. DE [15]', 'MAP', 'DE[3]']
        ylim = (0.86, 0.92)
    else:
        raise ValueError('Unknown use case')

    return {
        'ylim': ylim,
        'ylim_loss': ylim_loss,
        'baseline_acc': baseline_acc,
        'baseline_loss': baseline_loss,
        'baseline_name': baseline_name
    }


def get_x_ticks(num_ticks):
    """
    Get the x-ticks for a plot with a given number of ticks.
    :param num_ticks: Number of ticks
    :return: List of ticks
    """
    if num_ticks >= 200:
        every_ = 50
    elif num_ticks >= 100:
        every_ = 20
    elif num_ticks >= 50:
        every_ = 10
    elif num_ticks >= 20:
        every_ = 2
    else:
        every_ = 1
    return every_

def plot_lr_loss(outpath, only_first=False, figsize=(5, 3)):
    """
    Plot the learning rate and loss for each training run.
    :param outpath: Path to the folder containing the scores files
    :param only_first: If True, only plot the first scores file
    :param figsize: Figure size
    """
    # Find all folders that contain a scores file
    scores_files = []
    for root, subdirs, files in os.walk(outpath):
        for file in files:
            if file.endswith('scores.json'):
                scores_files.append(os.path.join(root, file))
    scores_files = sorted(scores_files)

    # If only_first is True, only plot the first scores file
    if only_first:
        scores_files = [scores_files[0]]

    for scores_file in scores_files:
        outpath = os.path.dirname(scores_file) + '/'
        print(outpath)

        # Find scores file
        scores_file = [f for f in os.listdir(outpath) if f.endswith('scores.json')]
        if len(scores_file) == 0:
            continue
        else:
            scores_file = scores_file[0]
        with open(outpath + scores_file, 'r') as f:
            scores = json.load(f)

        # Get learning rate and loss
        loss = scores['history']['loss']
        if 'lr' in scores['history']:
            lr = scores['history']['lr']
        elif 'learning_rate' in scores['history']:
            lr = scores['history']['learning_rate']
        else:
            lr = np.ones(len(loss))
        lr = np.array(lr)
        epochs = range(1, len(loss) + 1)
        every_ = get_x_ticks(len(loss))

        to_plot = [
            ('Loss', 'loss'),
            ('Accuracy', 'accuracy')
        ]
        if 'auc' in scores['history']:
            to_plot.append(('AUC', 'auc'))
        if 'precision' in scores['history']:
            to_plot.append(('Precision', 'precision'))
            to_plot.append(('Recall', 'recall'))

        for (name, hist_name) in to_plot:
            lower_name = name.lower()

            fig, ax = plt.subplots(1, 1, figsize=figsize)
            plt.setp(ax.spines.values(), color='#DDDDDD')

            ax.grid(which='major', color='#DDDDDD', linewidth=0.8, zorder=0)
            # Show the minor grid as well. Style it in very light gray as a thin, dotted line.
            #ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
            # Make the minor ticks and gridlines show.
            #ax.minorticks_on()
            ax.set_xticks(np.arange(1, len(loss) + 1, 1))
            # Labels: Every xth label is shown, starting from the xth label
            labels = [""] * len(loss)
            for i in range(1, len(loss) + 1):
                if i % every_ == 0:
                    labels[i - 1] = str(i)
            ax.set_xticklabels(labels)
            ax.set_xlim(0, len(loss) + 1)

            ax.plot(epochs, scores['history'][hist_name], label=f'Train {name}', color='darkolivegreen')
            ax.plot(epochs, scores['history']['val_' + hist_name], label=f'Val. {name}', color='darkorange')
            # Scale the learning rate to start at max of this hist_name
            scale = 1 / max(lr)
            ax.plot(epochs, scale * lr, label='lr (scaled)', c=(0.1, 0.1, 0.1, 0.5))
            #ax.set_title(f'Train and Validation {name}')
            ax.set_xlabel('Epochs')
            ax.set_ylabel(name)

            min_ = min(np.min(scores['history'][hist_name]), np.min(scores['history']['val_' + hist_name]), np.min(scale * lr))
            max_ = max(np.max(scores['history'][hist_name]), np.max(scores['history']['val_' + hist_name]))
            y_lim = (0.9 * min_, 1.1 * max_)
            plt.ylim(y_lim)
            plt.xlim(1, len(loss))
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
            ax.tick_params(color='#DDDDDD', which='both')
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.2,
                             box.width, box.height * 0.9])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                      fancybox=True, shadow=True, ncol=3)
            plt.savefig(outpath + f'{lower_name}.pdf', bbox_inches="tight")
            plt.close()


def plot_pac_bayes(outpath):
    """
    Plot the accuracy and weights for each ensemble member.
    :param outpath: Path to the folder containing the pac-bayes folder
    """
    # Find all pac-bayes folders
    pac_bayes_folders = []
    for root, subdirs, files in os.walk(outpath):
        if 'pac-bayes' in subdirs:
            pac_bayes_folders.append(os.path.join(root, 'pac-bayes/'))
    pac_bayes_folders = sorted(pac_bayes_folders)

    for outpath in pac_bayes_folders:
        # Load the distribution of weights and accuracies
        rhos_file = [f for f in os.listdir(outpath) if f.endswith('rhos.csv')][0]
        risks_file = [f for f in os.listdir(outpath) if f.endswith('iRProp.csv')][0]
        df_rhos = pd.read_csv(outpath + rhos_file, sep=';')
        df_risks = pd.read_csv(outpath + risks_file, sep=';')
        #print(outpath + rhos_file)
        #print(outpath + risks_file)
        every_ = get_x_ticks(len(df_rhos['h']))

        #print("number of val points: ", df_risks['n_min'][0])
        #print("number of tandem val points: ", df_risks['n2_min'][0])

        # 3 subplots
        fig, ax = plt.subplots(3, 1, figsize=(5, 3))
        for a in ax:
            plt.setp(a.spines.values(), color='#DDDDDD')
            a.grid(which='major', color='#EEEEEE', linewidth=0.8, zorder=0)
        # Plot the accuracy for each member in subplot 1 as bar plot without spacing between the bars
        ax[0].bar(df_rhos['h'], 1 - df_rhos['risk'], label='Accuracy', color='#222222', width=1.05, zorder=3)
        # Plot the weights
        ax[1].bar(df_rhos['h'], df_rhos['rho_lam'], label='FO', color=colors['uniform_last_per_model'], zorder=3)
        ax[2].bar(df_rhos['h'], df_rhos['rho_mv2'], label='TND', color=colors['tnd_last_per_model'], zorder=3)

        # x axis label for last subplot
        ax[2].set_xlabel('M')
        ax[0].set_ylabel(r'$\hat{A}$')
        ax[1].set_ylabel('ρ')
        ax[2].set_ylabel('ρ')

        for a in ax:
            a.set_xticks(np.arange(1, len(df_rhos['h']) + 1, 1))
            # Labels: Every xth label is shown, starting from the xth label
            labels = [""] * len(df_rhos['h'])
            for i in range(1, len(df_rhos['h']) + 1):
                if i % every_ == 0:
                    labels[i - 1] = str(i)
            a.set_xticklabels(labels)
            a.tick_params(color='#DDDDDD', which='both')
            a.set_xlim(0.5, len(df_rhos['h']) + 0.5)

        #plt.suptitle('Accuracy and weight distribution for ensemble')
        plt.tight_layout()
        plt.savefig(outpath + 'risk_weights.pdf')
        plt.close()


def plot_performances(folder, m, use_case, model_type):
    """
    Plot the performances of the different ensemble methods.
    """

    if 'epoch_budget' in folder:
        results_dict = {}
        for i in range(2, m+1):
            print(f'Ensemble size: {i}')
            folder_ = os.path.join(folder, f'{i:02d}')
            models = 0
            # Get number of models (in case not all are present)
            for root, dirs, files in os.walk(folder_):
                for file in files:
                    if file.endswith('scores.json'):
                        models += 1
            if not os.path.exists(os.path.join(folder_, 'ensemble_results.pkl')):
                print(f'File {os.path.join(folder_, "ensemble_results.pkl")} does not exist. Have you run ensemble prediction?')
                sys.exit(1)
            with open(os.path.join(folder_, 'ensemble_results.pkl'), 'rb') as f:
                results = pickle.load(f)
            results = results['results']
            for category in results.keys():
                if category not in results_dict:
                    results_dict[category] = ([], [], [], [], [], [])

                # Take last value of each metric (only biggest ensemble computed per epoch budget step)
                (mean, std, l_mean, l_std, div_mean, div_std) = results[category]
                results_dict[category][0].append((i, mean[-1][1], mean[-1][2]))
                results_dict[category][1].append((i, std[-1][1], std[-1][2]))
                results_dict[category][2].append((i, l_mean[-1][1]))
                results_dict[category][3].append((i, l_std[-1][1]))
                results_dict[category][4].append((i, div_mean[-1][1]))
                results_dict[category][5].append((i, div_std[-1][1]))
        categories = results_dict.keys()
        best_single_model_accuracy = None
        best_single_model_loss = None
        results = results_dict
    else:
        if not os.path.exists(os.path.join(folder, 'ensemble_results.pkl')):
            print(f'Path {os.path.join(folder, "ensemble_results.pkl")} does not exist. Have you run ensemble prediction?')
            sys.exit(1)
        with open(os.path.join(folder, 'ensemble_results.pkl'), 'rb') as f:
            results = pickle.load(f)
        categories = results['results'].keys()
        best_single_model_accuracy = results['best_single_model_accuracy']
        best_single_model_loss = results['best_single_model_loss']
        results = results['results']

    every_ = get_x_ticks(m)
    use_case_data = get_use_case_data(use_case, model_type)

    for f, figsize in [(os.path.join(folder, 'large'), (9, 6)), (os.path.join(folder, 'small'), (6, 4))]:
        os.makedirs(f, exist_ok=True)
        for plot_majority_vote, single_model_reference, mean_idx, std_idx, y_label, ylim_name, baseline_name, filename in [
            (False, best_single_model_accuracy, 0, 1, '$\hat{A}$', 'ylim', 'baseline_acc', 'ensemble_accs'),
            (True, best_single_model_accuracy, 0, 1, '$\hat{A}$', 'ylim', 'baseline_acc', 'ensemble_accs_majority_vote'),
            (False, best_single_model_loss, 2, 3, '$\hat{L}_{CE}$', 'ylim_loss', 'baseline_loss', 'ensemble_losses')
        ]:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            plt.setp(ax.spines.values(), color='#DDDDDD')
            ax.grid(which='major', color='#DDDDDD', linewidth=0.8, zorder=0)
            #ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
            #ax.minorticks_on()

            min_ = single_model_reference
            max_ = single_model_reference
            for category in categories:
                ensemble_mean, ensemble_std = results[category][mean_idx], results[category][std_idx]
                ensemble_mean = np.array(ensemble_mean)
                ensemble_std = np.array(ensemble_std)

                if min_ is None or min_ > np.min(ensemble_mean[:, -1]):
                    min_ = np.min(ensemble_mean[:, -1])
                if max_ is None or max_ < np.max(ensemble_mean[:, -1]):
                    max_ = np.max(ensemble_mean[:, -1])

                # Get a color
                color = colors[category]
                if plot_majority_vote:
                    plt.plot(ensemble_mean[:, 0], ensemble_mean[:, 2], label=f'{display_categories[category]}',color=color)
                    plt.plot(ensemble_mean[:, 0], ensemble_mean[:, 1], label=f'{display_categories[category].replace("AVG", "MV")}', color=color, linestyle='--')
                    # Std as area around the mean
                    plt.fill_between(ensemble_mean[:, 0], ensemble_mean[:, 1] - ensemble_std[:, 1],
                                     ensemble_mean[:, 1] + ensemble_std[:, 1], alpha=0.3, color=color, zorder=3)
                    plt.fill_between(ensemble_mean[:, 0], ensemble_mean[:, 2] - ensemble_std[:, 2],
                                     ensemble_mean[:, 2] + ensemble_std[:, 2], alpha=0.3, color=color, zorder=3)
                else:
                    plt.plot(ensemble_mean[:, 0], ensemble_mean[:, -1],
                             label=f'{display_categories[category]}', color=color)
                    # Std as area around the mean (Last element is the softmax average)
                    plt.fill_between(ensemble_mean[:, 0], ensemble_mean[:, -1] - ensemble_std[:, -1],
                                     ensemble_mean[:, -1] + ensemble_std[:, -1], alpha=0.3, color=color, zorder=3)
            plt.xlabel('M')
            plt.ylabel(y_label)
            print(min_, max_)
            if np.isfinite(max_):
                ylim = (max(min_ - 0.005, use_case_data[ylim_name][0]), max(max_ + 0.005, use_case_data[ylim_name][1]))
            else:
                ylim = (max(min_ - 0.005, use_case_data[ylim_name][0]), use_case_data[ylim_name][1])
            plt.ylim(ylim)
            # Horizontal line for the accuracy of the best model
            if single_model_reference is not None:
                plt.axhline(single_model_reference, color='orange', linestyle='--', label='Best SGD')
            if use_case_data[baseline_name] is not None:
                # Horizontal line for baseline accuracy
                for i, (acc, name) in enumerate(zip(use_case_data[baseline_name], use_case_data['baseline_name'])):
                    # Get a linestyle
                    linestyle = linestyles[(i % len(linestyles))]
                    # Get a color
                    greyscale = int(255 * i/len(use_case_data[baseline_name]))
                    color = f'#{greyscale:02x}{greyscale:02x}{greyscale:02x}'
                    plt.axhline(acc, label=name, color=color, linestyle=linestyle)
            ax.set_xticks(np.arange(1, m + 1, 1))
            # Labels: Every xth label is shown, starting from the xth label
            labels = [""] * m
            for i in range(1, m + 1):
                if i % every_ == 0:
                    labels[i - 1] = str(i)
            ax.set_xticklabels(labels)
            plt.xlim(2, m)
            ax.tick_params(color='#DDDDDD', which='both')
            # Shrink current axis's height by 10% on the bottom and move 20% up
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.2,
                             box.width, box.height * 0.9])

            # Put a legend below current axis
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                      fancybox=True, shadow=True, ncol=4)
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax.text(0.02, 0.97, f"{use_case_display[use_case]} {model_type}", transform=ax.transAxes, fontsize=9,
                    verticalalignment='center', bbox=props)
            plt.savefig(os.path.join(f, filename + ".pdf"), bbox_inches="tight")
            plt.close()


def create_table_results(path, m, use_case, model_type):
    """
    Automatically create a table with the results for the different ensemble methods.
    :param path: Path to the folder containing the experiments (per dataset and model)
    :param epoch_budget_folder: Name of the ensemble size for the epoch budget experiments
    :param use_case: Name of the use case. Can be 'cifar10', 'cifar100', 'imdb', 'retinopathy'
    :param model_type: Name of the model. Can be 'ResNet20v1', 'ResNet110v1', 'WideResNet28-10'
    """
    use_case_data = get_use_case_data(use_case, model_type)
    # Get subdirectories
    experiments = [f.path for f in os.scandir(path) if f.is_dir() and 'old' not in f.name]
    round_digits = 3

    display_name_order = {
        'original': (0, 'Original'),
        'sse': (3, 'SSE'),
        'bootstr': (2, 'Bagging'),
        'original_checkpointing': (1, 'Checkpointing'),
        'epoch_budget': (4, '')
    }
    experiments = sorted(experiments, key=lambda x: display_name_order[os.path.basename(x)][0])

    table_bounds = []
    table_performances = []
    for e in experiments:
        print(e)
        if 'epoch_budget' in e:
            # Find highest performing ensemble size
            # Get all ensemble_results.pkl files recursively
            ensemble_results_files = []
            for root, subdirs, files in os.walk(e):
                for file in files:
                    if file == 'ensemble_results.pkl':
                        ensemble_results_files.append(os.path.join(root, file))
            ensemble_results_files = sorted(ensemble_results_files)
            # Get the highest performing ensemble size
            max_acc = 0
            max_idx = 0
            for ensemble_results_file in ensemble_results_files:
                with open(ensemble_results_file, 'rb') as f:
                    ensemble_results = pickle.load(f)
                    max_ = max(ensemble_results['results']['uniform_last_per_model'][0][-1][2],
                                  ensemble_results['results']['tnd_last_per_model'][0][-1][2])
                    if max_ > max_acc:
                        max_acc = max_
                        max_idx = ensemble_results['results']['uniform_last_per_model'][0][-1][0]
            exp_name = f'Ep.b. ()'
            print(f"Best Epoch budget ensemble size: {max_idx}")
            m = max_idx
            e = os.path.join(e, f'{max_idx:02d}')
        else:
            exp_name = display_name_order[os.path.basename(e)][1]
        # Get the pac-bayes folder
        pac_bayes = [f.path for f in os.scandir(e) if f.name == 'pac-bayes'][0]
        # Load the data
        risks_file = [f for f in os.listdir(pac_bayes) if f.endswith('iRProp.csv')][0]
        df_risks = pd.read_csv(os.path.join(pac_bayes, risks_file), sep=';')
        # Open ensemble_results.pkl
        with open(os.path.join(e, 'ensemble_results.pkl'), 'rb') as f:
            ensemble_results = pickle.load(f)
            if m is not None:
                # Index for m either m-2 or the number of models in the ensemble
                max_m_to_evaluate = min(m-2, len(ensemble_results['results']['uniform_last_per_model'][0])-1)
            else:
                max_m_to_evaluate = len(ensemble_results['results']['uniform_last_per_model'][0])-1
            # Softmax average
            uni_last_sa = (round(ensemble_results['results']['uniform_last_per_model'][0][max_m_to_evaluate][2], round_digits),
                           ensemble_results['results']['uniform_last_per_model'][0][max_m_to_evaluate][0])
            tnd_last_sa = (round(ensemble_results['results']['tnd_last_per_model'][0][max_m_to_evaluate][2], round_digits),
                           ensemble_results['results']['tnd_last_per_model'][0][max_m_to_evaluate][0])
            # Majority vote
            tnd_last_mv = (round(ensemble_results['results']['tnd_last_per_model'][0][max_m_to_evaluate][1], round_digits),
                           ensemble_results['results']['tnd_last_per_model'][0][max_m_to_evaluate][0])
            if any(['all_per_model' in k for k in ensemble_results['results'].keys()]):
                uni_all_sa = (round(ensemble_results['results']['uniform_all_per_model'][0][max_m_to_evaluate][2], round_digits),
                              ensemble_results['results']['uniform_all_per_model'][0][max_m_to_evaluate][0])
                tnd_all_sa = (round(ensemble_results['results']['tnd_all_per_model'][0][max_m_to_evaluate][2], round_digits),
                              ensemble_results['results']['tnd_all_per_model'][0][max_m_to_evaluate][0])
                tnd_all_mv = (round(ensemble_results['results']['tnd_all_per_model'][0][max_m_to_evaluate][1], round_digits),
                              ensemble_results['results']['tnd_all_per_model'][0][max_m_to_evaluate][0])

                # Get the maximum of the last and all performances for TND
                max_tnd_sa = (max(tnd_last_sa[0], tnd_all_sa[0]), tnd_all_sa[1])
                max_tnd_mv = (max(tnd_last_mv[0], tnd_all_mv[0]), tnd_all_mv[1])

                performances_str = (f"{uni_last_sa[0]}[{uni_last_sa[1]}] & "
                                    f"{tnd_last_sa[0]}[{tnd_last_sa[1]}] & "
                                    f"{uni_all_sa[0]}[{uni_all_sa[1]}] & "
                                    f"{tnd_all_sa[0]}[{tnd_all_sa[1]}]")
            else:
                max_tnd_sa = tnd_last_sa
                max_tnd_mv = tnd_last_mv
                performances_str = f"{uni_last_sa[0]}[{uni_last_sa[1]}] & {tnd_last_sa[0]}[{tnd_last_sa[1]}] & - & -"

            bayesian_ref = f"{use_case_data['baseline_acc'][0]}"
            # Find ensemble size as [0-9] in baseline name, e.g. [10]
            ensemble_size = re.findall(r"\[\d+\]", use_case_data['baseline_name'][0])
            if len(ensemble_size) > 0:
                ensemble_size = ensemble_size[0][1:-1]
                bayesian_ref += f"[{ensemble_size}]"

            unf_bound = round(1 - df_risks['unf_tnd'][0], round_digits)
            tnd_bound = round(1 - df_risks['tnd_tnd'][0], round_digits)

        table_bounds.append(f"{exp_name} & {unf_bound} & {tnd_bound} & {max_tnd_sa[0]} & {max_tnd_mv[0]} \\\\")
        table_performances.append(
            f"{exp_name} & {performances_str} & {round(ensemble_results['best_single_model_accuracy'], round_digits)} & {bayesian_ref} \\\\")

    print("Bounds:")
    for t in table_bounds:
        print(t)

    print("Performances:")
    for t in table_performances:
        print(t)


def main():
    parser = argparse.ArgumentParser(description='Plotting')
    parser.add_argument('--path', type=str, help='Path to the folder containing the experiments')
    parser.add_argument('--epoch_budget_folder', type=str,
                        help='Name of the ensemble size for the epoch budget experiments')
    parser.add_argument('--lr_loss', action='store_true', help='Plot the learning rate and loss')
    parser.add_argument('-m', '--num_ensemble_members', type=int, help='Number of ensemble members to plot')
    parser.add_argument('--performances', action='store_true', help='Plot the performances')
    parser.add_argument('--pac_bayes', action='store_true', help='Plot the PAC-Bayes bounds')
    parser.add_argument('--table', action='store_true', help='Create a table with the results')
    parser.add_argument('--only_first', action='store_true', help='Only plot the first scores file')

    args = parser.parse_args()

    for root, dirs, files in os.walk(args.path):
        for file in files:
            if file.endswith('config.json'):
                with open(os.path.join(root, file), 'r') as f:
                    config = json.load(f)
                    if 'use_case' not in config:
                        if 'imdb' in args.path:
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
    print(f'Use case: {use_case}, model: {model_type}')

    if args.lr_loss:
        plot_lr_loss(args.path, args.only_first)
    if args.pac_bayes:
        plot_pac_bayes(args.path)
    if args.performances:
        plot_performances(args.path, args.m, use_case, model_type)
    if args.table:
        create_table_results(args.path, args.m, use_case, model_type)


if __name__ == '__main__':
    main()
    