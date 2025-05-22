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
    file_names = ['ensemble_accs.pdf', 'ensemble_aucs.pdf', 'rhos.pdf', 'ensemble_accs_majority_vote.pdf', 'ensemble_losses.pdf', 'loss.pdf', 'accuracy.pdf']
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
            if any([file_name in file for file_name in file_names]) and all([folder not in root for folder in unconsidered_folders]):
                pdf_files.append(os.path.join(root, file))

    print(len(pdf_files))
    for pdf_file in pdf_files:
        relative_path = os.path.relpath(pdf_file, path)
        os.makedirs(os.path.join(tmp_folder, os.path.dirname(relative_path)), exist_ok=True)
        shutil.copy(pdf_file, os.path.join(tmp_folder, relative_path))


if __name__ == '__main__':
    path = r"C:\Users\NHaup\OneDrive\Dokumente\Master_Studium\Semester_4\Thesis\Results"

    create_plots_folder(path)
    #print_accuracies()
    #sort_epoch_budget_folders(path)

    #path = r"C:\Users\NHaup\OneDrive\Dokumente\Master_Studium\Semester_4\Thesis\Results\imdb\original_checkpointing_same_val_data"
    # Open ensemble_results.pkl
    #import pickle
    #with open(os.path.join(path, 'ensemble_results.pkl'), 'rb') as f:
    #    ensemble_results = pickle.load(f)

    # Remove the tnd_last_per_model from the results
    #ensemble_results['results'].pop('tnd_last_per_model')

    # Save Copy
    #with open(os.path.join(path, 'ensemble_results.pkl'), 'wb') as f:
    #    pickle.dump(ensemble_results, f)

