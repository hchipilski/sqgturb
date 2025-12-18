# run_loc_smcmc_sqg.py
"""
Launcher for SQG-based Local SMCMC filter.

Reads model/obs settings from NetCDF produced by your SQG nature-run script,
sets up parameters and x_star (initial condition) from the saved NetCDF, then
launches loc_smcmc_sqg.loc_smcmc_filter simulations (parallel or serial).
"""

import os
import shutil
import time
from joblib import Parallel, delayed

import yaml
from netCDF4 import Dataset
import numpy as np

from loc_smcmc_sqg import loc_smcmc_filter, partition_domain
# If you prefer, you can also import get_params from your parameters module:
# from parameters import get_params


def load_sqg_params_from_ncs(params, nature_run_file_path, obs_file_path):
    """
    Read SQG model and obs settings from the saved NetCDF files and update params in place.
    - nature_run_file_path: path to 'sqg_lsmcmc_nature_....nc' (contains model attributes + grid dims)
    - obs_file_path: path to 'sqg_lsmcmc_obs_....nc' (contains obs attributes)
    This function mutates the params dict and returns it.
    """
    if not os.path.exists(nature_run_file_path):
        raise FileNotFoundError(f"nature_run_file file not found: {nature_run_file_path}")
    if not os.path.exists(obs_file_path):
        raise FileNotFoundError(f"obs_file file not found: {obs_file_path}")

    # Read model/nature-run file
    with Dataset(nature_run_file_path, mode='r') as nc:
        # dimensions
        if 'x' not in nc.dimensions or 'y' not in nc.dimensions or 'z' not in nc.dimensions:
            raise RuntimeError("nature_run_file NetCDF missing required dimensions 'x','y' or 'z'.")

        nx = len(nc.dimensions['x'])
        ny = len(nc.dimensions['y'])
        nlevs = len(nc.dimensions['z'])

        params['dgx'] = int(nx)
        params['dgy'] = int(ny)
        params['nlevs'] = int(nlevs)
        params['dim2'] = int(nx * ny)
        params['dimx'] = int(nlevs * nx * ny)

        # read common model attributes (if present)
        # timesteps in your saved file is model.timesteps -> assimilation interval
        if hasattr(nc, 'timesteps'):
            try:
                params['assim_timesteps'] = int(getattr(nc, 'timesteps'))
                params['t_freq'] = params['assim_timesteps']
                params['nassim'] = params['T']/params['t_freq']
            except Exception:
                pass

        for attr in ('dt', 'nsq', 'f', 'U', 'H', 'r', 'tdiab', 'diff_order',
                        'symmetric', 'diff_efold', 'theta0', 'g', 'sig_x'):
            if hasattr(nc, attr):
                val = getattr(nc, attr)
                # cast where sensible
                try:
                    params[attr] = float(val)
                except Exception:
                    params[attr] = val

        # read pv at t=0 for x_star (if variable 'pv' exists)
        if 'pv' in nc.variables:
            # pv has shape (t, z, y, x); we want pv[0] which is (z, y, x)
            pv0 = np.array(nc.variables['pv'][0], copy=True)
            # flatten in the order (levels, y, x)
            params['x_star'] = pv0.reshape(-1).astype(np.float32)
        else:
            # fallback: leave x_star to be filled from data_tools.get_data later
            params.setdefault('x_star', None)

    # Read the observation file (metadata)
    with Dataset(obs_file_path, mode='r') as nco:
        # read simple attributes if present
        if hasattr(nco, 'sig_y'):
            try:
                params['sig_y'] = float(getattr(nco, 'sig_y'))
            except Exception:
                params['sig_y'] = getattr(nco, 'sig_y')

        if hasattr(nco, 'swath_width'):
            params['swath_width'] = float(getattr(nco, 'swath_width'))
        if hasattr(nco, 'gap_width'):
            params['gap_width'] = float(getattr(nco, 'gap_width'))

        # we do not load yobs arrays here (loc_smcmc_sqg already reads them),
        # but we confirm the file exists and has expected dims/vars.
        # If desired we could preload the arrays into params (not done here).

    return params


def main():
    # directory setup (identical behaviour to your previous script)
    dir_ = "example_lsmcmc"
    path_ = "./example_lsmcmc"
    if os.path.isdir(path_):
        answer = input("Going to remove the current directory (%s). Do you want to remove the directory? [y]. To change its name enter [n]. " % path_)
        if answer.lower() in ["y", "yes"]:
            shutil.rmtree(dir_)
        elif answer.lower() in ["n", "no"]:
            print("You need to change the name of the directory %s. " % dir_)
            answer = input("Please enter a new name: ")
            os.rename(dir_, answer)

    dir_ = os.path.abspath(dir_)
    os.makedirs(dir_, exist_ok=True)
    print("Loading parameters from example_input_sqg.yml...")
    input_file = "example_input_sqg.yml"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"YAML parameter file not found: {input_file}")

    with open(input_file, 'r') as f:
        params = yaml.safe_load(f)

    # if YAML didn't specify nsimu, ncores, etc., ensure defaults (you can tune)
    params.setdefault('nsimu', int(params.get('nsimu', 1)))
    params.setdefault('ncores', int(params.get('ncores', 1)))
    params.setdefault('run_simuls_in_parallel', bool(params.get('run_simuls_in_parallel', False)))
    params.setdefault('nforecast', int(params.get('nforecast', 30)))
    params.setdefault('mcmc_N', int(params.get('mcmc_N', 5000)))
    params.setdefault('burn_in', int(params.get('burn_in', 1000)))
    # num_subdomains -> partitioning; keep both possible names
    params.setdefault('num_subdomains', int(params.get('num_subdomains', params.get('num_subdomains', 0))))

    # If using SQG-based assimilation, read model/obs settings from NetCDF files
    if params.get("use_SQG", True):
        nature_run_file = params.get('nature_run_file', None)
        obs_file = params.get('obs_file', None)
        if nature_run_file is None or obs_file is None:
            raise RuntimeError("When use_SQG=True you must provide 'nature_run_file' and 'obs_file' in the YAML input.")
        print(f"Reading SQG model/obs settings from:\n  nature_run_file: {nature_run_file}\n  obs_file:  {obs_file}")
        params = load_sqg_params_from_ncs(params, nature_run_file, obs_file)
        print("Injected model parameters from NetCDF into params dict.")
    else:
        print("use_SQG is False: using parameters as read from YAML.")

    # If x_star not set from nature_run_file NC, fallback to get_data
    if params.get('x_star') is None:
        # get_data signature: get_data(params, nout, "signal") used elsewhere
        print("x_star not found in nature_run_file NetCDF, calling get_data(...) for initial signal")
        params['x_star'] = np.asarray(get_data(params, 0, "signal"), dtype=np.float32)

    # Build partition labels. Your loc code expects `partition` (flat) and sometimes `partition_labels`.
    num_subdomains = int(params.get('num_subdomains', 0))
    if num_subdomains > 0:
        # partition_domain returns (partitions, partition_labels, N, nby, nbx, bh, bw)
        partitions, partition_labels, Nblocks, nby, nbx, bh, bw = partition_domain(params['dgy'], params['dgx'], N=num_subdomains)
    else:
        # choose a default partitioning based on params['num_subdomains'] or fallback
        partitions, partition_labels, Nblocks, nby, nbx, bh, bw = partition_domain(params['dgy'], params['dgx'], N=params.get('num_subdomains', None))

    params["partition_labels"] = partition_labels
    #params["partition"] = partition_labels.reshape(-1)  # flat representation
    params["num_subdomains"] = int(Nblocks)

    # place x_star into params as requested: pv[0] flattened already set by load_sqg_params_from_ncs
    # ensure shape matches dimx
    x_star = np.asarray(params['x_star'], dtype=np.float32)
    if x_star.size != int(params['dimx']):
        # maybe pv[0] was missing; try get_data fallback
        print("Warning: x_star length does not match computed dimx. Falling back to get_data(...).")
        x_star = np.asarray(get_data(params, 0, "signal"), dtype=np.float32)
        if x_star.size != int(params['dimx']):
            raise RuntimeError("x_star size mismatch and fallback failed: {} != {}".format(x_star.size, params['dimx']))

    params['x_star'] = x_star

    # small delay before launching
    time.sleep(1)

    simu = range(0, params["nsimu"])
    starttime = time.time()

    if params["run_simuls_in_parallel"]:
        print("Performing %d simulations on %d processors" % (params["nsimu"], params["ncores"]))
        Parallel(n_jobs=params["ncores"])(delayed(loc_smcmc_filter)(params, h) for h in simu)
    else:
        for h in simu:
            print("Performing simulation %d on 1 processor" % h)
            loc_smcmc_filter(params, h)

    endtime = time.time() - starttime
    print("Finished SMCMC Filtering in ", endtime, "seconds")


if __name__ == "__main__":
    main()
