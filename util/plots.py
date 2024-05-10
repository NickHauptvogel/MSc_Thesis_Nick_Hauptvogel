import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
import numpy as np
import pandas as pd
import os
import re
import pickle

from ensemble_prediction import get_use_case_data


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
        # Scale the learning rate to start at 1
        scale = 1 / np.max(lr)
        epochs = range(1, len(loss) + 1)
        every_ = get_x_ticks(len(loss))

        to_plot = [
            ('Loss', ['loss'], (0, 1.1 * max(loss)), 'upper right'),
            ('Accuracy', ['accuracy'], (0, 1), 'lower right'),
        ]
        if 'auc' in scores['history']:
            to_plot.append(('AUC', ['auc'], (0, 1), 'lower right'))
        if 'precision' in scores['history']:
            to_plot.append(('Precision', ['precision'], (0, 1), 'lower right'))
            to_plot.append(('Recall', ['recall'], (0, 1), 'lower right'))

        for (name, hist_names, ylim, legend_loc) in to_plot:
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

            for n in hist_names:
                ax.plot(epochs, scores['history'][n], label=f'Training {name}', color='darkolivegreen')
                ax.plot(epochs, scores['history']['val_' + n], label=f'Validation {name}', color='darkorange')
            ax.plot(epochs, scale * lr, label='Learning rate (scaled)', c=(0.1, 0.1, 0.1, 0.5))
            ax.set_title(f'Training and Validation {name}')
            ax.set_xlabel('Epochs')
            ax.set_ylabel(name)
            ax.legend(loc=legend_loc)
            plt.ylim(ylim)
            plt.xlim(1, len(loss))
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
            ax.tick_params(color='#DDDDDD', which='both')
            plt.tight_layout()
            plt.savefig(outpath + f'{lower_name}.pdf')
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
        print(outpath + rhos_file)
        print(outpath + risks_file)
        every_ = get_x_ticks(len(df_rhos['h']))

        print("number of val points: ", df_risks['n_min'][0])
        print("number of tandem val points: ", df_risks['n2_min'][0])

        # 3 subplots
        fig, ax = plt.subplots(3, 1, figsize=(5, 3))
        for a in ax:
            plt.setp(a.spines.values(), color='#DDDDDD')
            a.grid(which='major', color='#EEEEEE', linewidth=0.8, zorder=0)
        # Plot the accuracy for each member in subplot 1 as bar plot without spacing between the bars
        ax[0].bar(df_rhos['h'], 1 - df_rhos['risk'], label='Accuracy', color='#222222', width=1.05, zorder=3)
        # Plot the weights
        ax[1].bar(df_rhos['h'], df_rhos['rho_lam'], label='FO', color='tab:orange', zorder=3)
        ax[2].bar(df_rhos['h'], df_rhos['rho_mv2'], label='TND', color='tab:green', zorder=3)

        # x axis label for last subplot
        ax[2].set_xlabel('Ensemble member')
        ax[0].set_ylabel('Accuracy')
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

        plt.suptitle('Accuracy and weight distribution for ensemble')
        plt.tight_layout()
        plt.savefig(outpath + 'risk_weights.pdf')
        plt.close()


def create_table_results(path, epoch_budget_folder, use_case, model_type):
    """
    Automatically create a table with the results for the different ensemble methods.
    :param path: Path to the folder containing the experiments (per dataset and model)
    :param epoch_budget_folder: Name of the ensemble size for the epoch budget experiments
    :param use_case: Name of the use case. Can be 'cifar10', 'cifar100', 'imdb', 'retinopathy'
    """
    use_case_data = get_use_case_data(use_case, model_type)
    # Get subdirectories
    experiments = [f.path for f in os.scandir(path) if f.is_dir()]
    round_digits = 3

    display_name_order = {
        'original': (0, 'Original'),
        'sse': (3, 'SSE'),
        'bootstr': (2, 'Bagging'),
        'original_checkpointing': (1, 'Checkpointing'),
        'epoch_budget': (4, '')
    }
    experiments = sorted(experiments, key=lambda x: display_name_order[os.path.basename(x)][0])

    all_per_model = False
    table_bounds = []
    table_performances = []
    for e in experiments:
        if 'epoch_budget' in e:
            e = os.path.join(e, epoch_budget_folder)
            exp_name = f'Ep.b. ()'
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
            # Softmax average
            uni_last_sa = (round(ensemble_results['results']['uniform_last_per_model'][0][-1][2], round_digits),
                           ensemble_results['results']['uniform_last_per_model'][0][-1][0])
            tnd_last_sa = (round(ensemble_results['results']['tnd_last_per_model'][0][-1][2], round_digits),
                           ensemble_results['results']['tnd_last_per_model'][0][-1][0])
            # Majority vote
            tnd_last_mv = (round(ensemble_results['results']['tnd_last_per_model'][0][-1][1], round_digits),
                           ensemble_results['results']['tnd_last_per_model'][1][-1][0])
            if any(['all_per_model' in k for k in ensemble_results['results'].keys()]):
                all_per_model = True
                uni_all_sa = (round(ensemble_results['results']['uniform_all_per_model'][0][-1][2], round_digits),
                              ensemble_results['results']['uniform_all_per_model'][0][-1][0])
                tnd_all_sa = (round(ensemble_results['results']['tnd_all_per_model'][0][-1][2], round_digits),
                              ensemble_results['results']['tnd_all_per_model'][0][-1][0])
                tnd_all_mv = (round(ensemble_results['results']['tnd_all_per_model'][0][-1][1], round_digits),
                              ensemble_results['results']['tnd_all_per_model'][1][-1][0])

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

                performances_str = f"{uni_last_sa[0]}[{uni_last_sa[1]}] & {tnd_last_sa[0]}[{tnd_last_sa[1]}]"
                if all_per_model:
                    performances_str += " & &"

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plotting')
    parser.add_argument('--path', type=str, help='Path to the folder containing the experiments')
    parser.add_argument('--epoch_budget_folder', type=str, help='Name of the ensemble size for the epoch budget experiments')
    parser.add_argument('--use_case', type=str, help='Name of the use case. Can be "cifar10", "cifar100", "imdb", "retinopathy"')
    parser.add_argument('--model_type', type=str, help='Name of the model type. Can be "ResNet20v1", "ResNet110v1", "WideResNet28-10"')
    parser.add_argument('--lr_loss', action='store_true', help='Plot the learning rate and loss')
    parser.add_argument('--pac_bayes', action='store_true', help='Plot the PAC-Bayes bounds')
    parser.add_argument('--table', action='store_true', help='Create a table with the results')
    parser.add_argument('--only_first', action='store_true', help='Only plot the first scores file')

    args = parser.parse_args()

    if args.lr_loss:
        plot_lr_loss(args.path, args.only_first)
    if args.pac_bayes:
        plot_pac_bayes(args.path)
    if args.table:
        create_table_results(args.path, args.epoch_budget_folder, args.use_case, args.model_type)
