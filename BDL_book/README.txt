This is the code accompanying the chapter "PAC-Bayesian deep neural network ensembles", especially the practical
exercise 5. It contains the implementation of the optimization of the PAC-Bayesian bound for deep neural network
ensembles using pre-computed predictions on the IMDb dataset (from independent training runs).

How to use:

1) Make sure you have Python 3 installed, along with the required packages listed in requirements.txt.
   You can install the required packages using pip:
   ```
   pip install -r requirements.txt
   ```

2) Run the script `ensemble_plot_imdb.py` to optimize the PAC-Bayesian bound and generate the plots:
   ```
   python ensemble_plot_imdb.py
   ```