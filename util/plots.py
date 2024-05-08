import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
import numpy as np
import pandas as pd
import os
import re
import pickle

from ensemble_prediction import get_use_case_data

def plot_lr_loss(outpath, only_first=False):

    # Find all folders that contain a scores file
    scores_files = []
    for root, subdirs, files in os.walk(outpath):
        for file in files:
            if file.endswith('scores.json'):
                scores_files.append(os.path.join(root, file))
    scores_files = sorted(scores_files)

    if only_first:
        scores_files = [scores_files[0]]

    figsize = (6, 4)


    for scores_file in scores_files:
        outpath = os.path.dirname(scores_file) + '/'
        print(outpath)

        # Find scores file
        scores_file = [f for f in os.listdir(outpath) if f.endswith('scores.json')]
        if len(scores_file) == 0:
            continue
        else:
            scores_file = scores_file[0]
        with open(outpath+scores_file, 'r') as f:
            scores = json.load(f)

        loss = scores['history']['loss']
        if 'lr' in scores['history']:
            lr = scores['history']['lr']
        elif 'learning_rate' in scores['history']:
            lr = scores['history']['learning_rate']
        else:
            lr = np.ones(len(loss))
        lr = np.array(lr)
        scale = 1/np.max(lr)
        epochs = range(1, len(loss) + 1)

        if len(loss) >= 200:
            every_ = 20
        elif len(loss) >= 100:
            every_ = 10
        elif len(loss) >= 50:
            every_ = 5
        elif len(loss) >= 20:
            every_ = 2
        else:
            every_ = 1

        to_plot = [
            ('Loss', ['loss'], (0, 1.1*max(loss)), 'upper right'),
            ('Accuracy', ['accuracy'], (0, 1), 'lower left'),
        ]
        if 'auc' in scores['history']:
            to_plot.append(('AUC', ['auc'], (0, 1), 'lower left'))
        if 'precision' in scores['history']:
            to_plot.append(('Precision', ['precision'], (0, 1), 'lower left'))
            to_plot.append(('Recall', ['recall'], (0, 1), 'lower left'))

        for (name, hist_names, ylim, legend_loc) in to_plot:
            lower_name = name.lower()

            fig, ax = plt.subplots(1, 1, figsize=figsize)
            plt.setp(ax.spines.values(), color='#DDDDDD')

            ax.grid(which='major', color='#DDDDDD', linewidth=0.8, zorder=0)
            # Show the minor grid as well. Style it in very light gray as a thin,
            # dotted line.
            ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
            # Make the minor ticks and gridlines show.
            ax.minorticks_on()
            ax.set_xticks(np.arange(1, len(loss) + 1, 1))
            # Labels: Every xth label is shown, starting from the xth label
            labels = [""] * len(loss)
            for i in range(1, len(loss) + 1):
                if i % every_ == 0:
                    labels[i - 1] = str(i)
            ax.set_xticklabels(labels)
            ax.set_xlim(0, len(loss) + 1)

            for n in hist_names:
                ax.plot(epochs, scores['history'][n], label=f'Training {name}')
                ax.plot(epochs, scores['history']['val_'+n], label=f'Validation {name}')
            ax.plot(epochs, scale*lr, label='Learning rate (scaled)', c=(0.1, 0.1, 0.1, 0.5))
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

    # Find all pac-bayes folders
    pac_bayes_folders = []
    for root, subdirs, files in os.walk(outpath):
        if 'pac-bayes' in subdirs:
            pac_bayes_folders.append(os.path.join(root, 'pac-bayes/'))
    pac_bayes_folders = sorted(pac_bayes_folders)

    for outpath in pac_bayes_folders:
        # Load the data
        rhos_file = [f for f in os.listdir(outpath) if f.endswith('rhos.csv')][0]
        risks_file = [f for f in os.listdir(outpath) if f.endswith('iRProp.csv')][0]
        df_rhos = pd.read_csv(outpath + rhos_file, sep=';')
        df_risks = pd.read_csv(outpath + risks_file, sep=';')
        print(outpath + rhos_file)
        print(outpath + risks_file)

        num_members = len(df_rhos['h'])
        if num_members > 50:
            every_ = 5
        elif num_members > 30:
            every_ = 2
        else:
            every_ = 1

        print("number of val points: ", df_risks['n_min'][0])
        print("number of tandem val points: ", df_risks['n2_min'][0])

        # 3 subplots
        fig, ax = plt.subplots(3, 1, figsize=(10, 6))
        for a in ax:
            plt.setp(a.spines.values(), color='#DDDDDD')
            a.grid(which='major', color='#EEEEEE', linewidth=0.8, zorder=0)
        # Plot the accuracy for each member in subplot 1 as bar plot without spacing between the bars
        ax[0].bar(df_rhos['h'], 1-df_rhos['risk'], label='Accuracy', color='#222222', width=1.05, zorder=3)
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
            # Labels: Every 5th label is shown, starting from the xth label
            labels = [""] * len(df_rhos['h'])
            for i in range(1, len(df_rhos['h']) + 1):
                if i % every_ == 0:
                    labels[i-1] = str(i)
            a.set_xticklabels(labels)
            a.tick_params(color='#DDDDDD', which='both')
            a.set_xlim(1, len(df_rhos['h']))
            a.legend(loc='upper left')

        plt.suptitle('Accuracy and weight distribution for ensemble')
        plt.tight_layout()
        plt.savefig(outpath + 'risk_weights.pdf')
        plt.close()

        # Plot the bounds for the different ensemble types
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        plt.setp(ax.spines.values(), color='#DDDDDD')
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8, zorder=0)
        # Show the minor grid as well. Style it in very light gray as a thin,
        # dotted line.
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
        # Make the minor ticks and gridlines show.
        ax.minorticks_on()
        # Plot the bounds for the different ensemble types next to each other
        dist_ = 0.21
        width_ = 0.4
        ax.bar([1 - dist_, 2 - dist_], [1-df_risks['unf_pbkl'][0], 1-df_risks['unf_tnd'][0]],
               label='Before optim.', color=(0.6, 0.2, 0.2, 0.5), width=width_, edgecolor=(0.6, 0.2, 0.2, 1.), linewidth=1,
               zorder=3)
        ax.bar([1 + dist_, 2 + dist_], [1-df_risks['lam_pbkl'][0], 1-df_risks['tnd_tnd'][0]],
               label='After optim.', color=(0.2, 0.6, 0.2, 0.5), width=width_, edgecolor=(0.2, 0.6, 0.2, 1.), linewidth=1,
               zorder=3)

        min_y = 1-max([df_risks['unf_pbkl'][0], df_risks['lam_pbkl'][0], df_risks['unf_tnd'][0], df_risks['tnd_tnd'][0]])
        max_y = 1-min([df_risks['unf_pbkl'][0], df_risks['lam_pbkl'][0], df_risks['unf_tnd'][0], df_risks['tnd_tnd'][0]])
        ax.set_ylim(max(0, min_y - 0.1), min(1, max_y + 0.1))

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['First order', 'Tandem loss'])
        ax.tick_params(color='#DDDDDD', which='both')
        ax.set_xlabel('Weighting scheme')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy bounds of different ensemble weightings')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(outpath + 'risk_bounds.pdf')
        plt.close()

        # Plot the risks for the different ensemble types
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        plt.setp(ax.spines.values(), color='#DDDDDD')
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8, zorder=0)
        # Show the minor grid as well. Style it in very light gray as a thin,
        # dotted line.
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
        # Make the minor ticks and gridlines show.
        ax.minorticks_on()
        # Plot the bounds for the different ensemble types next to each other
        ax.bar([1 + 0.1, 2 + 0.1, 3 + 0.1],
               [1-df_risks['unf_mv_risk_softmax_avg'][0], 1-df_risks['lam_mv_risk_softmax_avg'][0],
                1-df_risks['tnd_mv_risk_softmax_avg'][0]],
               label='Softmax average', color=(0.2, 0.6, 0.2, 0.5), width=0.17, edgecolor=(0.2, 0.6, 0.2, 1.),
               linewidth=1,
               zorder=3)
        ax.bar([1 - 0.1, 2 - 0.1, 3 - 0.1], [1-df_risks['unf_mv_risk_maj_vote'][0], 1-df_risks['lam_mv_risk_maj_vote'][0],
                                             1-df_risks['tnd_mv_risk_maj_vote'][0]],
               label='Majority vote', color=(0., 0., 1., 0.5), width=0.17, edgecolor=(0., 0., 1., 1.), linewidth=1,
               zorder=3)

        ax.set_ylim(0.5, 1.0)
        min_uniform = min([df_risks['unf_mv_risk_softmax_avg'][0], df_risks['unf_mv_risk_maj_vote'][0]])
        min_lambda = min([df_risks['lam_mv_risk_softmax_avg'][0], df_risks['lam_mv_risk_maj_vote'][0]])
        min_tandem = min([df_risks['tnd_mv_risk_softmax_avg'][0], df_risks['tnd_mv_risk_maj_vote'][0]])
        print("minimal risk uniform: ", min_uniform, " (accuracy: ", 1 - min_uniform, ")")
        print("minimal risk lambda: ", min_lambda, " (accuracy: ", 1 - min_lambda, ")")
        print("minimal risk tandem: ", min_tandem, " (accuracy: ", 1 - min_tandem, ")")
        # Label bars with accuracies
        ax.text(1.15, 1 - df_risks['unf_mv_risk_softmax_avg'][0], f"{1-df_risks['unf_mv_risk_softmax_avg'][0]:.3f}", ha='center', va='bottom')
        ax.text(2.15, 1 - df_risks['lam_mv_risk_softmax_avg'][0], f"{1-df_risks['lam_mv_risk_softmax_avg'][0]:.3f}", ha='center', va='bottom')
        ax.text(3.15, 1 - df_risks['tnd_mv_risk_softmax_avg'][0], f"{1-df_risks['tnd_mv_risk_softmax_avg'][0]:.3f}", ha='center', va='bottom')
        ax.text(0.85, 1 - df_risks['unf_mv_risk_maj_vote'][0], f"{1-df_risks['unf_mv_risk_maj_vote'][0]:.3f}", ha='center', va='bottom')
        ax.text(1.85, 1 - df_risks['lam_mv_risk_maj_vote'][0], f"{1-df_risks['lam_mv_risk_maj_vote'][0]:.3f}", ha='center', va='bottom')
        ax.text(2.85, 1 - df_risks['tnd_mv_risk_maj_vote'][0], f"{1-df_risks['tnd_mv_risk_maj_vote'][0]:.3f}", ha='center', va='bottom')


        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Uniform', 'First Order', 'Tandem'])
        ax.tick_params(color='#DDDDDD', which='both')
        ax.set_xlabel('Weighting scheme')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy of different ensemble weightings')
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(outpath + 'risks.pdf')
        plt.close()


def create_table_results(path, epoch_budget_folder, use_case, model_type):
    use_case_data = get_use_case_data(use_case, model_type)
    # Get subdirectories
    experiments = [f.path for f in os.scandir(path) if f.is_dir()]

    display_name_order = {
        'original': (0, 'Original'),
        'sse': (3, 'SSE'),
        'bootstr': (2, 'Bagging'),
        'original_checkpointing': (1, 'Original (checkp.)'),
        'epoch_budget': (4, '')
    }
    experiments = sorted(experiments, key=lambda x: display_name_order[os.path.basename(x)][0])

    all_per_model = False
    for e in experiments:
        if 'epoch_budget' in e:
            e = os.path.join(e, epoch_budget_folder)
            exp_name = f'Ep.b. ({os.path.basename(e)})'
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
            uni_last = f"{round(ensemble_results['results']['uniform_last_per_model'][0][-1][-1], 4)}[{ensemble_results['results']['uniform_last_per_model'][0][-1][0]}]"
            tnd_last = f"{round(ensemble_results['results']['tnd_last_per_model'][0][-1][-1], 4)}[{ensemble_results['results']['tnd_last_per_model'][0][-1][0]}]"
            if any(['all_per_model' in k for k in ensemble_results['results'].keys()]):
                all_per_model = True
                uni_all = f"{round(ensemble_results['results']['uniform_all_per_model'][0][-1][-1], 4)}[{ensemble_results['results']['uniform_all_per_model'][0][-1][0]}]"
                tnd_all = f"{round(ensemble_results['results']['tnd_all_per_model'][0][-1][-1], 4)}[{ensemble_results['results']['tnd_all_per_model'][0][-1][0]}]"
                perf_res = f"{uni_last} & {tnd_last} & {uni_all} & {tnd_all}"
            else:
                perf_res = f"{uni_last} & {tnd_last}"
                if all_per_model:
                    perf_res += " & &"

            bayesian_ref = f"{use_case_data['baseline_acc'][0]}"
            # Find ensemble size as [0-9] in baseline name, e.g. [10]
            ensemble_size = re.findall(r"\[\d+\]", use_case_data['baseline_name'][0])
            if len(ensemble_size) > 0:
                ensemble_size = ensemble_size[0][1:-1]
                bayesian_ref += f"[{ensemble_size}]"

        print(f"{exp_name} & {round(1-df_risks['unf_tnd'][0], 4)} & {round(1-df_risks['tnd_tnd'][0], 4)} & {perf_res} & {round(ensemble_results['best_single_model_accuracy'], 4)} & {bayesian_ref} \\\\")


if __name__ == '__main__':
    #path = 'results/cifar10/resnet110/'
    path = r"C:\Users\NHaup\OneDrive\Dokumente\Master_Studium\Semester_4\Thesis\Results\imdb"
    use_case = 'imdb'
    model_type = 'ResNet110v1'
    create_table_results(path, '03', use_case, model_type)
    #plot_lr_loss(path, only_first=True)
    #plot_pac_bayes(path)
