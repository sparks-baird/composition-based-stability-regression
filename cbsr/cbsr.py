from matbench.bench import MatbenchBenchmark
from crabnet._crabnet import CrabNet
mb = MatbenchBenchmark(subset=["matbench_mp_e_form"])
task = list(mb.tasks)[0]
task.load()

# %% load the matbench data for formation energy

# %% load the most recent snapshot of Materials Project formation energy, `e_above_hull`, and mpids using MPRester()
# make sure to not post your API key by doing e.g. https://pymatgen.org/usage.html#setting-the-pmg-mapi-key-in-the-config-file
# https://github.com/sparks-baird/mat_discover/blob/main/mat_discover/utils/generate_elasticity_data.py
# https://github.com/sparks-baird/RoboCrab/blob/master/download-stable-elasticity.py

for fold in task.folds:
    train_inputs, train_outputs = task.get_train_and_val_data(fold)

    # calculate avg, min, max, and std formation energies for repeat compositions, using e.g. a modified version of groupby_formula
    # https://github.com/sparks-baird/mat_discover/blob/b92501384865bfe455a7d186487c972bec0a01b0/mat_discover/utils/data.py#L8

    # cross-reference the Materials Project snapshot MPIDs with the Matbench MPIDs to add e_above_hull to Matbench data

    # %% format training and validation data as DataFrames with "formula", "avg", "min", "max", "std", and "e_above_hull" columns
    form_name = "formula"
    targ_name = "target"
    avg_name = "avg"
    min_name = "min"
    max_name = "max"
    std_name = "std"

    eform_names = [avg_name, min_name, max_name, std_name]

    ehull_name = "e_above_hull"

    avg_sig_name, min_sig_name, max_sig_name, std_sig_name = [name + "_sigma" for name in [avg_name, min_name, max_name, std_name]]
    sigma_names = [avg_sig_name, min_sig_name, max_sig_name, std_sig_name]

    extended_names = eform_names + sigma_names

    # %% naive, robust transfer-learning-like implementation using multiple CrabNet models
    # https://crabnet.readthedocs.io/en/latest/crabnet.html#crabnet.crabnet_.CrabNet

    epochs = 300
    # Ax/SAASBO hyperparameter optimization results could be integrated https://arxiv.org/abs/2203.12597

    # formation energy models and e_above_hull model
    # TODO: (assuming preliminary results are "good") save each of the models
    cb_avg = CrabNet(epochs=epochs)
    cb_min = CrabNet(epochs=epochs)
    cb_max = CrabNet(epochs=epochs)
    cb_std = CrabNet(epochs=epochs)

    cb_hull = CrabNet(epochs=epochs, extend_features=extended_names)

    # fit the extended features to use for transfer learning
    # I don't think the extra columns will cause issues, otherwise only take necessary via e.g. train_df[[form_name, avg_name]]
    cb_avg.fit(train_df.rename(columns={avg_name: targ_name})
    cb_min.fit(train_df.rename(columns={min_name: targ_name})
    cb_max.fit(train_df.rename(columns={max_name: targ_name})
    cb_std.fit(train_df.rename(columns={std_name: targ_name})
            
    test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
            
    val_df[avg_name], val_df[avg_sig_name] = cb_avg.predict(val_df, return_uncertainty=True)
    val_df[min_name], val_df[min_sig_name] = cb_min.predict(val_df, return_uncertainty=True)
    val_df[max_name], val_df[max_sig_name] = cb_max.predict(val_df, return_uncertainty=True)
    val_df[std_name], val_df[std_sig_name] = cb_std.predict(val_df, return_uncertainty=True)           

    cb_hull.fit(train_df.rename(columns={ehull_name: targ_name})
    ehull_pred, ehull_sigma, ehull_true = cb_hull.predict(val_df, return_uncertainty=True, return_true=True)
                
    # %% also compute convex hull directly from min formation energy predictions
