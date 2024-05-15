import os
import json

def sort_epoch_budget_folders():
    """
    Sort the epoch budget single run folders into ensemble folders.
    Example:
        20240101_120000_03_05/ -> 03/05/
    """
    target_folder = '../../Results/cifar100/wideresnet2810/epoch_budget'

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


def print_accuracies():
    path = '../../Results/cifar100/resnet110/epoch_budget'
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


def create_plots_zip():
    import shutil
    paths = [r'C:\Users\NHaup\Projects\Results',
             r'C:\Users\NHaup\OneDrive\Dokumente\Master_Studium\Semester_4\Thesis\Results']

    # Create a tmp folder that copies all .pdf files
    tmp_folder = '../plots'
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)
    for path in paths:
        # Find all .pdf files recursively
        pdf_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('ensemble_accs.pdf') or file.endswith('risk_weights.pdf'):
                    pdf_files.append(os.path.join(root, file))

        print(len(pdf_files))
        for pdf_file in pdf_files:
            relative_path = os.path.relpath(pdf_file, path)
            os.makedirs(os.path.join(tmp_folder, os.path.dirname(relative_path)), exist_ok=True)
            shutil.copy(pdf_file, os.path.join(tmp_folder, relative_path))


if __name__ == '__main__':
    create_plots_zip()
    #print_accuracies()
    #sort_epoch_budget_folders()
