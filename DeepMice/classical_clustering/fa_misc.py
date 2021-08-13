# Imports
import numpy as np
import os
import pandas as pd
from scipy.stats import variation
from tqdm import tqdm
import warnings


def smooth_func(y, box_pts):
    """
    Source: https://stackoverflow.com/a/26337730
    Parameters
    ----------
    y
    box_pts
    Returns
    -------
    """
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def fa_get_par_dirs(store_dir, absolute_tags, relative_tags):
    """ Function to select parameter directories based on par_tags.
    Parameters should have all absolute tags, and at least one of the
    relative tags.
    Parameters
    ----------
    store_dir : str
        Full path to directory with parameter directories
    absolute_tags : list of str(s) (optional)
        Every selected parameter should have all of these tags
    relative_tags : list of str(s) (optional)
        Every selected parameter should have at least one of these tags
    Returns
    -------
    par_names : list of str(s)
        Selected parameter names
    par_dirs : list of str(s)
        Full path of selected parameter directories
    """
    if not os.path.exists(store_dir):
        raise Warning("fa_get_par_dirs(): directory does not exist:",store_dir)

    # Loop over (parameter) directories, skip all other files
    par_names = next(os.walk(store_dir))[1]
    if absolute_tags:
        # Select par_names that have ALL absolute_tags
        if not isinstance(absolute_tags, list):
            absolute_tags = [absolute_tags]
        for par_tag in absolute_tags:
            par_names = [s for s in par_names if par_tag in s]

    if relative_tags:
        # Select par_names that have at least one of these tags
        if not isinstance(relative_tags, list):
            relative_tags =[relative_tags]
        for par_name in par_names:
            if not any(tag in par_name for tag in relative_tags):
                par_names.remove(par_name)

    if np.size(par_names) == 0:
        raise Warning("No parameters selected for absolute_tags {}. Please "
                      "change absolute_tags.".format(absolute_tags))
    # Get full paths to parameter directories
    par_dirs = [os.path.join(store_dir, par_name) for par_name in par_names]

    # Check if directory is empty
    for par_dir in par_dirs:
        if len(os.listdir(par_dir)) == 0:
            par_dirs.remove(par_dir)
            warnings.warn("fa_get_par_dirs(): Directory is empty, and "
                          "therefore removed from list:" + par_dir)
    return par_names, par_dirs


def fa_plot_list_features(f_parameter):
    """ Function to list all features and compartments of interest, as given in
     f_parameter. This also determines subplot size and sorting of features
     and compartments.
    Parameters
    ----------
    f_parameter : dict
        Describes features types and compartments of interest
    Returns
    -------
    f_type_list : list
        List with all features of interest
    compartment_list : list
        List with all compartments of interest
    """
    f_type_list = []
    compartment_list = []
    for f_par_value in f_parameter.values():
        f_type = f_par_value["type"]
        if f_type not in f_type_list:
            f_type_list.append(f_type)
        if 'compartment' in f_par_value:
            compartment = f_par_value["compartment"]
        else:
            compartment = 0
        if compartment not in compartment_list:
            compartment_list.append(compartment)
    f_type_list = sorted(f_type_list)

    if "speed" in f_type_list:
        f_type_list.remove("speed")
        f_type_list.insert(0, "speed")
    elif "attenuation" in f_type_list:
        f_type_list.remove("speed")
        f_type_list.insert(0, "speed")

    compartment_list = sorted(compartment_list)
    return f_type_list, compartment_list


def fa_plot_combine_features(par_dir, f_type_list, compartment_list):
    """ Function to gather all feature values from disk, and append them to a
    single list per parameter/feature.
    Parameters
    ----------
    par_dir : str
        Full list to parameter directory
    f_type_list : list of str(s)
        List of all features of interest
    compartment_list : list of int(s)
        List of all compartments of interest
    Returns
    -------
    f_values_dict : dict
        Dictionary containing list of all feature values of interest
        key : feature type (str)
        value : list of feature values
    feature_dict : dict
        Dictionary containing all features of interest
    factors : list of float(s)
        Parameter multiplication factors
    segment : str(s)
        Segment of parameter. Is "" when parameter has no specific segment
        assigned
    """
    # Select file paths
    file_paths = [os.path.join(par_dir, file_name) for file_name in
                  os.listdir(par_dir)]
    if np.size(file_paths) == 0:
        raise Warning("fa_plot_combine_features(): No files found in", par_dir)
    # Allocate variables
    f_values_dict = dict()
    factors = []
    segment = ""
    for file_path in file_paths:
        # Load data
        feature_dict = pd.read_hdf(file_path, key='features')
        param = pd.read_hdf(file_path, key='param')
        if 'segment' in param:
            segment = param['segment'].values[0]
        factors.append(param['factor'].values[0])
        # Extract feature values
        for f_name, feature_dict_value in feature_dict.items():
            f_type = feature_dict_value["type"]
            f_value = feature_dict_value["value"]
            compartment = feature_dict_value["compartment"]
            # Select only features of interest
            if f_type in f_type_list and compartment in compartment_list:
                if f_name in f_values_dict:
                    # Gather feature value to dict
                    f_values_dict[f_name].append(f_value)
                else:
                    f_values_dict[f_name] = [f_value]
    return f_values_dict, feature_dict, factors, segment


# %% Select best feature-parameter relations
def create_feature_matrix(par_names, par_dirs, f_parameter):
    """Creates array containing all feature values, for all parameters and
    factors. Order in the array: features, factors, parameters
    Parameters
    ----------
    par_names : list of str(s)
        Names of parameters to include
    par_dirs : list of str(s)
        Paths to parameter directories of parameters to include
    f_parameter : dict
        Describes features (objectives) of optimization problem
    Returns
    -------
    f_matrix : array of shape (N_features, N_factors, N_parameters)
    """
    f_names = list(f_parameter.keys())
    f_type_list, compartment_list = fa_plot_list_features(f_parameter)

    f_matrix = []
    for par_name, par_dir in tqdm(zip(par_names, par_dirs), total=len(
            par_names), desc="f_matrix"):
        # Collect feature values
        f_values_dict, feature_dict, factors, _ = fa_plot_combine_features(
            par_dir, f_type_list, compartment_list)
        f_values_list = []
        for f_name in f_names:
            # Collect in order of f_names
            f_values_list.append(f_values_dict[f_name])
        f_values_array = np.asarray(f_values_list)
        # Sort features
        sort_ind = np.asarray(factors).argsort()
        f_values_sorted = f_values_array[:, sort_ind[::-1]]
        # Stack to f_array and change size if needed
        if np.size(f_matrix) != 0:
            while np.size(f_values_sorted, 1) < np.size(f_matrix, 1):
                f_values_sorted = np.hstack((f_values_sorted, np.nan *
                                np.empty((np.size(f_values_sorted, 0), 1))))
            f_matrix = np.dstack((f_matrix, f_values_sorted))
        else:
            f_matrix = f_values_sorted
    return f_matrix


def get_units_list(par_dir, f_parameter):
    """ Retrieve units of features from par_dir for features described in
    f_parameter.
    Parameters
    ----------
    par_dir : list of str(s)
        Path to parameter directories of parameters to include
    f_parameter : dict
        Describes features (objectives) of optimization problem
    Returns
    -------
    f_units_list : list of str(s)
        Describes units as stored in par_dir
    """
    f_names = list(f_parameter.keys())
    f_type_list, compartment_list = fa_plot_list_features(f_parameter)
    f_values_dict, feature_dict, factors, _ = fa_plot_combine_features(
        par_dir, f_type_list, compartment_list)

    f_units_list = []
    for f_name in f_names:
        f_units_list.append(feature_dict[f_name]['unit'])

    return f_units_list


def normalise_matrix(f_matrix, f_units_list):
    """Normalise matrix values to measurement units.
    Dimensions corresponding to frequency and time are converted to units of
    1 ms. Dimensions corresponding to voltage (potential) are converted to
    units of 100 mV.
    Parameters
    ----------
    f_matrix : array of shape (N_features, N_factors, N_parameters)
    f_units_list : list of str(s)
        Describes units as stored in par_dir
    Returns
    -------
    f_matrix_norm : array of shape (N_features, N_factors, N_parameters)
        Normalised to measurement units
    f_units_list : list of str(s)
        Describes units as stored in par_dir, updated to normalised units.
    """
    units = list(set(f_units_list))
    f_matrix_norm = np.copy(f_matrix)
    for unit in units:
        unit_indices = [i for i, x in enumerate(f_units_list) if x == unit]
        if unit == "Hz":
            # Convert from Hz to 1 ms
            f_matrix_norm[unit_indices, :, :] = np.divide(
                1, f_matrix[unit_indices, :, :],
                out=np.zeros_like(f_matrix_norm[unit_indices, :, :]),
                where=f_matrix_norm[unit_indices, :, :] != 0) * 1E3
            for unit_index in unit_indices:
                f_units_list[unit_index] = "ms"
        elif unit == "mV":
            # Convert from mV to 100 mV
            f_matrix_norm[unit_indices, :, :] = np.divide(
                f_matrix_norm[unit_indices, :, :], 100)
            for unit_index in unit_indices:
                f_units_list[unit_index] = "100 mV"
    return f_matrix_norm, f_units_list


def select_largest_elements(matrix, f_units_list, adaptation=True):
    """ Select largest elements within each group of the same units.
    Parameters
    ----------
    matrix : array of shape (N_features, N_factors, N_parameters)
    f_units_list : list of str(s)
        Describes units as stored in par_dir, updated to normalised units.
    Returns
    -------
    feature_indices : list of int(s)
        Integers referring to matrix entries with selected elements
    parameter_indices : list of int(s)
        Integers referring to matrix entries with selected parameters
    """
    units = list(set(f_units_list))
    feature_indices = []
    parameter_indices = []
    for unit in units:
        unit_indices = [i for i, x in enumerate(f_units_list) if x == unit]
        new_matrix = np.zeros_like(matrix)
        new_matrix[unit_indices, :] = matrix[unit_indices, :]
        # Select number of desired features per unit
        if adaptation:
            if unit == "100 mV":
                N = 2
            elif unit == "ms":
                N = 1
            elif unit == "":
                N = 1
            else:  # m/s, mV/um
                N = 1
        else:
            if unit == "100 mV":
                N = 2
            elif unit == "ms":
                N = 2
            elif unit == "":
                N = 0
            else:   # m/s, mV/um
                N = 1

        if len(unit_indices) < N:
            N = len(unit_indices)
        # Select largest elements
        for n in range(N):
            feature_ind, parameter_ind = np.unravel_index(np.nanargmax(
                new_matrix), np.shape(new_matrix))
            feature_indices.append(feature_ind)
            parameter_indices.append(parameter_ind)
            new_matrix[feature_ind, :] = np.nan  # exclude selected feature
            new_matrix[:, parameter_ind] = np.nan  # exclude selected parameter
            matrix[feature_ind, :] = np.nan  # exclude selected feature
            matrix[:, parameter_ind] = np.nan  # exclude selected parameter
    return feature_indices, parameter_indices


def select_best_features(store_dir, f_parameter, f_matrix=None,
                         adaptation=True, absolute_tags=None,
                         relative_tags=None, cv=False):
    """ Wrapper function to select features with highest variance (and lowest
    coefficient of variation) Calls among others create_feature_matrix() (
    optional), select_largest_elements()
    Parameters
    ----------
    store_dir : str
        Full path to directory in which the final result is stored
    f_parameter : dict
        Describes features types and compartments of interest
    f_matrix : array of shape (N_features, N_factors, N_parameters) (optional)
        If not given, created by calling create_feature_matrix()
    adaptation : bool (optional)
        Whether to include adaptation in the selection or not
    absolute_tags : list of str(s) (optional)
        Every selected parameter should have all of these tags
    relative_tags : list of str(s) (optional)
        Every selected parameter should have at least one of these tags
    cv : Bool (optional)
        Whether to select on basis of coefficient of variation in addition
        with variance.
    Returns
    -------
    feature_indices : list of int(s)
        Integers referring to matrix entries with selected elements
    parameter_indices : list of int(s)
        Integers referring to matrix entries with selected parameters
    """
    par_names, par_dirs = fa_get_par_dirs(store_dir, absolute_tags,
                                          relative_tags)
    f_names = list(f_parameter.keys())

    if f_matrix is None:
        f_matrix = create_feature_matrix(par_names, par_dirs, f_parameter)

    # Get units list
    f_units_list = get_units_list(par_dirs[0], f_parameter)
    # Normalise f_matrix
    f_matrix_norm, f_units_list = normalise_matrix(f_matrix, f_units_list)
    # Calculate variance
    var_matrix = np.std(f_matrix_norm, 1)

    if cv:
        # Exclude feature-parameter combinations with cv larger than 1
        cv_matrix = variation(np.copy(f_matrix_norm), axis=1)
        f_cv, p_cv = np.unravel_index(np.where(abs(cv_matrix) > 1),
                                      np.shape(cv_matrix))
        var_matrix[:, p_cv] = np.nan
        cv_matrix[:, p_cv] = np.nan
        # Create selection matrix on basis of variance and cv
        cv_matrix_pivot = np.nanmax(cv_matrix, 1)[:, None] - cv_matrix
        matrix = np.sqrt(var_matrix ** 2 + cv_matrix_pivot ** 2)
    else:
        matrix = var_matrix

    # Select largest elements
    feature_indices, parameter_indices = select_largest_elements(
        matrix=matrix, f_units_list=f_units_list, adaptation=adaptation)
    return feature_indices, parameter_indices