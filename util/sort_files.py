import os

def sort_epoch_budget_folders():
    """
    Sort the epoch budget single run folders into ensemble folders.
    Example:
        20240101_120000_03_05/ -> 03/05/
    """
    target_folder = 'results/cifar100/resnet110/epoch_budget'

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


if __name__ == '__main__':
    sort_epoch_budget_folders()
