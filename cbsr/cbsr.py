from matbench.bench import MatbenchBenchmark
from crabnet._crabnet import CrabNet
import pandas as pd
from mp_api import MPRester


key = 'YOUR_API_KEY'
mpr = MPRester()
mb = MatbenchBenchmark(subset=["matbench_mp_e_form"])
task = list(mb.tasks)[0]
task.load()
# %% load the matbench data for formation energy

# %% load the most recent snapshot of Materials Project formation energy, `e_above_hull`, and mpids using MPRester()
# make sure to not post your API key by doing e.g. https://pymatgen.org/usage.html#setting-the-pmg-mapi-key-in-the-config-file
# https://github.com/sparks-baird/mat_discover/blob/main/mat_discover/utils/generate_elasticity_data.py
# https://github.com/sparks-baird/RoboCrab/blob/master/download-stable-elasticity.py

with MPRester(api_key=key) as mpr:
    query = mpr.query(criteria, ['pretty_formula', 'reduced_cell_formula', 'material_id', 'formation_energy_per_atom', 'e_above_hull'])
    print(len(query))
    pretty_formula = [query[i]['pretty_formula'] for i in range(len(query))]
    cell_formula = [query[i]['reduced_cell_formula'] for i in range(len(query))]
    mat_id = [query[i]['material_id'] for i in range(len(query))]
    form_e = [query[i]['formation_energy_per_atom'] for i in range(len(query))]
    e_hull = [query[i]['e_above_hull'] for i in range(len(query))]

ugly_formula = [] # get formulas in the form of "Li2O1", useful for crabnet I guess
for material in cell_formula: # a material is a dictionary
    coeff = [int(material[element]) for element in material] # a stoichiometric coefficient is the value associated to a material-key
    els = [element for element in material]    
    formula_lst = [str(i) + str(j) for i, j in zip(els, coeff)]
    formula = ''.join(formula_lst)
    ugly_formula.append(formula)
pretty_formula_array = np.array(pretty_formula)
ugly_formula_array = np.array(ugly_formula)
mat_id_array = np.array(mat_id)
form_e_array = np.array(form_e)
e_hull_array = np.array(e_hull)
data = np.column_stack((pretty_formula_array, ugly_formula_array, mat_id_array, form_e_array, e_hull_array))
df = pd.DataFrame(data = data, index = None, columns = ['pretty_formula', 'ugly_formula', 'mat_id', 'formation_energy', 'e_above_hull'])
df['formation_energy'] = df['formation_energy'].astype("float64")
df['e_above_hull'] = df['e_above_hull'].astype("float64")
df.reset_index(inplace=True)
#%%
for fold in task.folds:
    train_inputs, train_outputs = task.get_train_and_val_data(fold)

    # make DataFrame

    # calculate avg, min, max, and std formation energies for repeat compositions, using e.g. a modified version of groupby_formula
    # https://github.com/sparks-baird/mat_discover/blob/b92501384865bfe455a7d186487c972bec0a01b0/mat_discover/utils/data.py#L8

    train_df = pd.DataFrame({})

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

    avg_sig_name, min_sig_name, max_sig_name, std_sig_name = [
        name + "_sigma" for name in [avg_name, min_name, max_name, std_name]
    ]
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
    cb_avg.fit(train_df.rename(columns={avg_name: targ_name}))
    cb_min.fit(train_df.rename(columns={min_name: targ_name}))
    cb_max.fit(train_df.rename(columns={max_name: targ_name}))
    cb_std.fit(train_df.rename(columns={std_name: targ_name}))

    test_inputs, test_outputs = task.get_test_data(fold, include_target=True)

    # make validation DataFrame
    val_df = pd.DataFrame({})

    # predict on extended features
    kwargs = dict(return_uncertainty=True)
    val_df[avg_name], val_df[avg_sig_name] = cb_avg.predict(val_df, **kwargs)
    val_df[min_name], val_df[min_sig_name] = cb_min.predict(val_df, **kwargs)
    val_df[max_name], val_df[max_sig_name] = cb_max.predict(val_df, **kwargs)
    val_df[std_name], val_df[std_sig_name] = cb_std.predict(val_df, **kwargs)

    cb_hull.fit(train_df.rename(columns={ehull_name: targ_name}))
    ehull_pred, ehull_sigma, ehull_true = cb_hull.predict(
        val_df, return_uncertainty=True, return_true=True
    )

    # %% also compute convex hull directly from min formation energy predictions
