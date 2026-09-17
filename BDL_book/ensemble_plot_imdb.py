#!/usr/bin/env python3
import numpy as np
from majority_vote_bounds import optimize_rho_oob
from joblib import delayed, Parallel
import matplotlib.pyplot as plt


# ENSEMBLE CODE
def nan_argmax(arr):
    # NOTE np.argmax will return 0 for [nan,nan]
    res = np.full(arr.shape[:-1], np.nan)
    idxs = ~np.isnan(arr).any(axis=-1)
    argmax = np.argmax(arr, axis=-1)
    res[idxs] = argmax[idxs]
    return res

def nan_acc(X, y):
    """Calculate accuracy ignoring NaN predictions."""
    idxs = ~np.isnan(X)
    return np.mean(X[idxs] == y[idxs])

def chkpnt_acc(chkpnt, val_labels):
    idxs = ~np.isnan(chkpnt).any(axis=-1)
    return np.mean(chkpnt[idxs].argmax(-1) == val_labels[idxs])


def ensemble_predictions_AVG(rho, predictions):
    pred = (predictions * rho[:, None, None]).sum(axis=0)
    pred = nan_argmax(pred)
    return pred


def make_rho_ensemble_oob(subset_idx, dataset):
    preds_test = dataset["predictions_test"][*subset_idx.T]
    labels_test = dataset["labels_test"]

    preds_val = dataset["predictions_validation"][*subset_idx.T]
    preds_val = nan_argmax(preds_val)
    labels_val = dataset["labels_validation"]

    bound, rho, lam = optimize_rho_oob(preds_val, labels_val)
    preds_ensemble_test = ensemble_predictions_AVG(rho, preds_test)

    if len(subset_idx) <= 100:
        # Find best checkpoint per selected run
        selected_runs = np.unique(subset_idx[:, 0])
        best_checkpoints = []

        for run in selected_runs:
            best_cp = val_scores[run].argmax()

            # Find at which index this checkpoint exists in the subset_idx array
            matches = np.where(
                (subset_idx[:, 0] == run) &
                (subset_idx[:, 1] == best_cp)
            )[0]
            best_checkpoints.append(matches[0])
        plot_rho_distribution(rho, best_checkpoints=best_checkpoints, output_path=f"rho_distribution_{len(subset_idx)}_members.png")

    return nan_acc(preds_ensemble_test, labels_test), bound


def make_uniform_ensemble(subset_idx, dataset):
    preds_test = dataset["predictions_test"][*subset_idx.T]
    labels_test = dataset["labels_test"]

    rho = np.ones(len(subset_idx)) / len(subset_idx)
    preds_ensemble_test = ensemble_predictions_AVG(rho, preds_test)

    return nan_acc(preds_ensemble_test, labels_test)


def plot_rho_distribution(rho, best_checkpoints=None, output_path="rho_distribution.png"):
    """
    Plot the distribution of rho values.
    :param rho: Array of rho values.
    :param best_checkpoints: list or array
        Indices of checkpoints with highest validation scores.
    :param output_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.setp(ax.spines.values(), color='#DDDDDD')
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    ax.grid(which='major', color='#EEEEEE', linewidth=0.8, zorder=0)

    # Create a bar plot for rho values
    indices = np.arange(len(rho))

    # Default color: black
    colors = ['black'] * len(rho)

    # Highlight best checkpoints
    if best_checkpoints is not None:
        for idx in best_checkpoints:
            colors[idx] = 'red'

    ax.bar(indices, rho, color=colors, width=1.05, zorder=3)

    # Adjust x-axis labels for readability
    ax.set_xticks(indices)
    step = max(1, len(rho) // 10)  # Show at most 10 labels
    ax.set_xticklabels(indices + 1, rotation=45)
    for i, label in enumerate(ax.get_xticklabels()):
        if i % step != 0:
            label.set_visible(False)

    ax.set_xlabel(r"Ensemble Members ($M$)")
    ax.set_ylabel(r"$\rho$")
    ax.set_title(r"PAC-Bayes ensemble $\rho$ Distribution")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# PLOT CODE
ds_npz = np.load("imdb_predictions.npz")
ds = dict([(key, ds_npz[key]) for key in ds_npz.keys()])
n_iter, n_checkpoints, n_examples, n_cats = ds["predictions_validation"].shape
X = np.arange(2,31)


val_scores = np.zeros((n_iter, n_checkpoints))
val_scores += np.array(
    Parallel(-1)(
        delayed(chkpnt_acc)(chkpnt, ds["labels_validation"])
        for chkpnt in ds["predictions_validation"] \
        .reshape(n_iter*n_checkpoints, n_examples, n_cats)
    )
).reshape(n_iter, n_checkpoints)

n_r = 5

rho_acc = np.zeros((n_r, len(X)))
uni_acc = np.zeros((n_r, len(X)))
ear_acc = np.zeros((n_r, len(X)))

for trial_idx in range(n_r):
    chkpnts = [np.random.choice(n_iter, i, replace=False) for i in X]
    subsets = [np.array([(c,i) for c in cs for i in range(n_checkpoints)])
               for cs in chkpnts]

    best_idxs = [np.array(list(zip(c, val_scores[c].argmax(axis=1)))) for c in chkpnts]
    
    rho_ = np.array(Parallel(-1)(delayed(make_rho_ensemble_oob)(subset, ds) for subset in subsets))
    rho_acc[trial_idx] += rho_[:, 0]
    uni_acc[trial_idx] += np.array(Parallel(-1)(delayed(make_uniform_ensemble)(subset, ds) for subset in subsets))
    ear_acc[trial_idx] += np.array(Parallel(-1)(delayed(make_uniform_ensemble)(b, ds) for b in best_idxs))

fig, axs = plt.subplots()
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

uni_µ = np.mean(uni_acc, axis=0)
plt.plot(X, uni_µ, label="Uniform ensemble $f_{\\mathrm{uniform}}^{\\mathrm{all}}$", color=colors[0], marker='*', lw=2, ls='--')

ear_µ = np.mean(ear_acc, axis=0)
plt.plot(X, np.mean(ear_acc, axis=0), label="Ensemble of best checkpoints $f_{\\mathrm{uniform}}^{\\mathrm{val}}$", color=colors[2], marker='x', lw=2, ls='-.')

rho_µ = np.mean(rho_acc, axis=0)
plt.plot(X, np.mean(rho_acc, axis=0), label="PAC-Bayes ensemble $f_\\rho^{\\mathrm{all}}$", color=colors[1], lw=2, marker='o',)


plt.xlabel("Training run number")
plt.grid()
plt.ylabel("Accuracy")
plt.title(f"Ensembles IMDB")
plt.legend()
plt.savefig(f"imdb_ens_same_run.png", dpi=600)
plt.close()
