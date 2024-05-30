import os
import json
import shutil


def sort_epoch_budget_folders(target_folder):
    """
    Sort the epoch budget single run folders into ensemble folders.
    Example:
        20240101_120000_03_05/ -> 03/05/
    """

    # Get all subdirectories in current_folder that start with 2024
    subdirs = [f.path for f in os.scandir(target_folder) if f.is_dir() and f.name.startswith('2024')]

    for subdir in subdirs:
        # Get the ensemble size from the subdirectory name
        ensemble = subdir.split('_')[-2]
        model = subdir.split('_')[-1]
        # Make sure ensemble folder exists in target_folder
        ensemble_folder = os.path.join(target_folder, ensemble)
        if not os.path.exists(ensemble_folder):
            os.makedirs(ensemble_folder)
        # Move the subdirectory to the ensemble folder
        os.rename(subdir, os.path.join(ensemble_folder, model))


def print_accuracies(path):
    """
    Print the test accuracies of all scores.json files that have at least one accuracy below 0.5
    (to find unconverged runs)
    """
    # Find all scores.json files recursively
    scores_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('scores.json'):
                scores_files.append(os.path.join(root, file))

    for scores_file in scores_files:
        with open(scores_file, 'r') as f:
            scores = json.load(f)
            if any([float(score) < 0.5 for score in scores['test_accuracy']]):
                print(scores_file + str(scores['test_accuracy']))


def create_plots_folder(path):
    """
    Create a folder with all .pdf files from the given path
    """
    file_names = ['ensemble_accs.pdf', 'ensemble_aucs.pdf', 'risk_weights.pdf', 'ensemble_accs_majority_vote.pdf', 'ensemble_losses.pdf', 'loss.pdf', 'accuracy.pdf']
    unconsidered_folders = ['large', 'old'] + [f'epoch_budget\\{i:02d}' for i in range(1, 25)]
    # Create a tmp folder that copies all .pdf files
    tmp_folder = '../plots'
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)
    # Find all .pdf files recursively
    pdf_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file in file_names and all([folder not in root for folder in unconsidered_folders]):
                pdf_files.append(os.path.join(root, file))

    print(len(pdf_files))
    for pdf_file in pdf_files:
        relative_path = os.path.relpath(pdf_file, path)
        os.makedirs(os.path.join(tmp_folder, os.path.dirname(relative_path)), exist_ok=True)
        shutil.copy(pdf_file, os.path.join(tmp_folder, relative_path))


if __name__ == '__main__':
    path = r"C:\Users\NHaup\OneDrive\Desktop\tmp2"
    import numpy as np
    best = 0
    best_temp = 0
    # For each file in directory
    for file in os.listdir(path):
        # Open the file
        accs = []
        with open(os.path.join(path, file), 'r') as f:
            # Read all lines
            lines = f.readlines()
            # Find Test set metrics: line
            for i, line in enumerate(lines):
                if 'Test set metrics:' in line:
                    # Get the line 19 lines below
                    accs.append(float(lines[i+19].split()[-1]))
                if "* * * Run sgmcmc for seed = 1" in line:
                    temp_str = line

            print(np.mean(accs))

            if np.mean(accs) > best:
                best = np.mean(accs)
                best_temp = temp_str
    print("Best accuracy:")
    print(best)
    print(best_temp)

    #create_plots_folder(path)
    #print_accuracies()
    #sort_epoch_budget_folders(path)
