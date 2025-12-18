# loc_smcmc_sqg.py
import os
import time
import numpy as np
import h5py

from data_tools import (get_data, get_data_info)
from scipy import linalg as scl

# netCDF reader/writer
from netCDF4 import Dataset
import tempfile
# SQG model (your implementation must be importable)
try:
    from sqgturb import SQG, rfft2, irfft2
except Exception as e:
    raise ImportError("sqgturb.SQG import failed: {}".format(e))


def loc_smcmc_filter(params, isim):
    """main function of MCMC filtering"""
    np.random.seed(isim)
    smcmc_f = Loc_SMCMC_Filter(isim, params)
    smcmc_f.run_main()


def get_divisors(n):
    return [i for i in range(1, n + 1) if n % i == 0]


def partition_domain(ny, nx, N=None, min_cells_per_block=6):
    total_cells = ny * nx

    if N is None:
        best = None
        max_blocks = 0
        for nby in get_divisors(ny):
            for nbx in get_divisors(nx):
                bh = ny // nby
                bw = nx // nbx
                cells_per_block = bh * bw
                if cells_per_block >= min_cells_per_block:
                    N_candidate = nby * nbx
                    if N_candidate > max_blocks:
                        best = (nby, nbx, bh, bw)
                        max_blocks = N_candidate
        if best is None:
            raise ValueError(f"No partition found with at least {min_cells_per_block} cells per block")
        nby, nbx, bh, bw = best
        N = nby * nbx

    else:
        if total_cells % N != 0:
            raise ValueError(f"N={N} must divide total number of cells {total_cells}")

        # Try to find the best (nby, nbx) giving near-square blocks
        candidates = []
        for nby in get_divisors(N):
            nbx = N // nby
            if ny % nby == 0 and nx % nbx == 0:
                bh = ny // nby
                bw = nx // nbx
                aspect_ratio = max(bh, bw) / min(bh, bw)
                candidates.append((aspect_ratio, abs(bh - bw), nby, nbx, bh, bw))

        if not candidates:
            raise ValueError(f"Cannot partition domain into {N} blocks of equal size")

        candidates.sort()  # prioritize square-like and balanced shapes
        _, _, nby, nbx, bh, bw = candidates[0]

    partitions = []
    for i in range(nby):
        for j in range(nbx):
            y_start = i * bh
            y_end = (i + 1) * bh
            x_start = j * bw
            x_end = (j + 1) * bw
            partitions.append(((y_start, y_end), (x_start, x_end)))

    partition_labels = np.empty((ny, nx), dtype=np.int32)
    for label, ((y_start, y_end), (x_start, x_end)) in enumerate(partitions):
        partition_labels[y_start:y_end, x_start:x_end] = label

    return partitions, partition_labels, N, nby, nbx, bh, bw


def block_diag_einsum(arr, num):
    rows, cols = arr.shape
    result = np.zeros((num, rows, num, cols), dtype=arr.dtype)
    diag = np.einsum('ijik->ijk', result)
    diag[:] = arr
    return result.reshape(rows * num, cols * num)


class Loc_SMCMC_Filter():
    """
    Local SMCMC filter adapted to use SQG forward model in place of linear A*x.
    MCMC sampling loops remain inline (as in your original code). Observations are
    read from an observation NetCDF file (params['obs_file']) when available.
    """
    def __init__(self, isim, params):
        self.params = params  # store early
        params["isim"] = isim
        params["time"] = 0.0

        # --- grid & dimensions (compute dimx inside code) ---
        self.nlevs = int(self.params.get("nlevs", 2))
        self.ny = int(self.params.get("dgy"))
        self.nx = int(self.params.get("dgx"))
        self.ncells = self.ny * self.nx
        self.params["dimx"] = int(self.nlevs * self.ncells)
        self.dimx = self.params["dimx"]

        # total model steps T (wallclock steps, includes non-assim steps)
        self.T = int(self.params["T"])

        # assimilation timesteps (how many model dt per assimilation)
        self.assim_timesteps = int(self.params.get("assim_timesteps", self.params.get("t_freq", 1)))
        self.params["t_freq"] = int(self.assim_timesteps)

        # number of assimilation cycles (floor)
        self.nassim = int(self.T // self.params["t_freq"])

        # main arrays: store only at assimilation times: index 0..nassim (0 = initial state)
        self.lsmcmc_mean = np.zeros((self.nassim + 1, self.dimx), dtype=np.float32)

        # MCMC iteration counts
        self.mcmc_iters = int(self.params["mcmc_N"] + self.params["burn_in"])
        self.forecast = np.zeros((self.dimx, int(self.params["nforecast"])), dtype=float)

        self.t_simul = 0.0
        self.tsteps = np.zeros(self.T, dtype=float)
        self.nstep = 0

        # process-noise flag and amplitude
        self.add_noise_sig_every_dt = bool(self.params.get("add_noise_sig_every_dt", False))
        self.sig_x = float(self.params.get("sig_x", 0.0))


        # minimal SQG constructor args extracted from params (pass only what's present)
        # keep threads from env if present
        self.threads = int(os.getenv('OMP_NUM_THREADS', '1'))
        # Build sqg_args from params with safe defaults where keys exist
        self.sqg_args = {}
        for key in ("nsq", "f", "U", "H", "r", "tdiab", "dt", "diff_order", "diff_efold",
                    "theta0", "g", "dealias", "symmetric", "threads", "precision", "tstart"):
            if key in self.params:
                self.sqg_args[key] = self.params[key]
                print(key, self.sqg_args[key])
        # ensure threads is present
        if "threads" not in self.sqg_args:
            self.sqg_args["threads"] = self.threads

        self.scalefact = self.params['f']*self.params['theta0']/self.params['g']
        # partition labels: prefer provided partition in params, otherwise create one
        if "partition" in self.params and self.params["partition"] is not None:
            part = np.array(self.params["partition"])
            if part.size == self.ny * self.nx:
                self.partition_labels = part.reshape((self.ny, self.nx)).astype(np.int32)
            else:
                self.partition_labels = part.astype(np.int32)
        else:
            nblocks = int(self.params.get("num_subdomains", 0))
            if nblocks > 0:
                _, partition_labels, _, _, _, _, _ = partition_domain(self.ny, self.nx, N=nblocks)
                self.partition_labels = partition_labels
            else:
                # default single block
                self.partition_labels = np.zeros((self.ny, self.nx), dtype=np.int32)

        # --- load observation NetCDF (if provided) ---
        self.yobs_all = None
        self.yobs_ind_all = None
        self.yobs_ind_level0_all = None
        self.n_obs_times = 0
        obs_file = self.params.get("obs_file", self.params.get("saveobs", self.params.get("data_info_file", None)))
        if obs_file is not None and os.path.exists(obs_file):
            try:
                nc = Dataset(obs_file, mode='r')
                if 'yobs_all' in nc.variables:
                    self.yobs_all = np.array(nc.variables['yobs_all'][:])
                if 'yobs_ind_all' in nc.variables:
                    self.yobs_ind_all = np.array(nc.variables['yobs_ind_all'][:], dtype=int)
                if 'yobs_ind_level0_all' in nc.variables:
                    self.yobs_ind_level0_all = np.array(nc.variables['yobs_ind_level0_all'][:], dtype=int)
                # number of obs time slices saved (these are saved at nout-1 in your generation code)
                if self.yobs_all is not None:
                    self.n_obs_times = self.yobs_all.shape[0]
                elif self.yobs_ind_all is not None:
                    self.n_obs_times = self.yobs_ind_all.shape[0]
                nc.close()
                self.obs_file = obs_file
            except Exception as e:
                print("Warning: failed to open obs file '{}': {}. Falling back to get_data/get_data_info".format(obs_file, e))
                self.obs_file = None
        else:
            self.obs_file = None

        # path to nature-run file (for truth); optionally provided in params
        self.nature_file = self.params.get("nature_run_file", self.params.get("savedata", None))

        with Dataset(self.nature_file, mode='r') as nc:
            self.obtimes = nc.variables['t'][:]
        self.sqg_args['tstart'] = self.obtimes[0]
        # allocate RMSE vector (one per assimilation cycle)
        self.RMSE = np.full((self.nassim,), np.nan, dtype=np.float32)

        # initial x_star placement if provided in params (pv[0] flattened)
        if 'x_star' in self.params and self.params['x_star'] is not None:
            xs = np.asarray(self.params['x_star'], dtype=np.float32)
            if xs.size != self.dimx:
                raise RuntimeError("params['x_star'] length != dimx")
            self.lsmcmc_mean[0] = xs
        else:
            # leave zeros for now; run_main will attempt to fill from nature file or get_data when available
            pass

        # persistent forecast models (list of SQG instances), initialized on run_main
        self.forecast_models = None
        # temporary container for the last observed level-0 indices (set during _get_obs_from_nc or fallback)
        self.indxob = np.array([], dtype=int)

    # -------------------------
    # small helpers
    # -------------------------
    def _flat_to_grid(self, vec_flat):
        return vec_flat.reshape((self.nlevs, self.ny, self.nx)).astype(np.float32)

    def _grid_to_flat(self, grid):
        return grid.ravel().astype(np.float32)

    # Build persistent forecast models (one instance per forecast member).
    def _init_forecast_models(self):
        """
        Initialize persistent SQG instances for each ensemble member from the initial state.
        Also initializes self.forecast columns from that initial state (plus optional initial noise).
        """
        Nf = int(self.params["nforecast"])
        # determine initial flat state: prefer provided lsmcmc_mean[:,0], else params['x_star'], else nature file pv[0]
        init_flat = None
        if np.any(self.lsmcmc_mean[0]):
            init_flat = self.lsmcmc_mean[0].copy()
        elif 'x_star' in self.params and self.params['x_star'] is not None:
            init_flat = np.asarray(self.params['x_star'], dtype=np.float32).copy()
 

        init_grid = self._flat_to_grid(init_flat)

        # create Nf SQG instances, each with its own copy of the grid
        self.forecast_models = []
        for j in range(Nf):
            # the SQG constructor expects a grid shaped pv (nlevs,ny,nx)
            model_j = SQG(init_grid.copy(), **self.sqg_args)
            # ensure model_j advances one dt per call; we will call advance() once per model dt
            model_j.timesteps = 1
            model_j.t = self.obtimes[0]
            model_j.pvspec = rfft2(init_grid, threads=self.threads)
            self.forecast_models.append(model_j)

        # initialize forecast array from init_flat
        self.forecast = np.tile(init_flat[:, None], (1, Nf)).astype(float)


    def _advance_forecast_one_step(self):
        """
        Advance each persistent forecast model by exactly one model timestep (dt),
        update self.forecast columns and update model.pvspec after adding physical-space noise.
        """
        Nf = len(self.forecast_models)
        for j in range(Nf):
            # model_j.timesteps already set to 1 at construction
            self.forecast_models[j].advance()  # returns physical-space PV array
            self.forecast_models[j].timesteps = 1
            pv_j = irfft2(self.forecast_models[j].pvspec)
            if self.add_noise_sig_every_dt and (self.sig_x > 0.0):
                pv_j += np.random.normal(scale=self.sig_x, size=pv_j.shape).astype(pv_j.dtype)
            self.forecast_models[j].pvspec = rfft2(pv_j, threads=self.threads)
            self.forecast[:, j] = self._grid_to_flat(pv_j)


    # -------------------------
    # likelihoods (same as your original)
    # -------------------------
    def logmvnpdf_g(self, X_minus_mu, sigd):
        xSigSqrtinv = X_minus_mu / sigd
        quadform = np.sum(xSigSqrtinv ** 2, axis=0)
        ly = -0.5 * quadform
        return ly

    def logmvnpdf_f_iidNoise(self, X_minus_mu, sigd):
        xSigSqrtinv = X_minus_mu / sigd
        return -0.5 * np.sum(xSigSqrtinv ** 2, axis=0)

    def logmvnpdf_f(self, sv_A, ind0, ind1, indy, X_minus_mu, N):
        A0 = X_minus_mu[sv_A]
        Q_inv_loc = self.params["Q_inv"][sv_A, :]
        Q_inv_loc = Q_inv_loc[:, sv_A]
        return -0.5 * np.dot(np.matmul(A0.T, Q_inv_loc), A0)

    def compute_pi_N(self, sv_A, ind0, ind1, indy, yn, zn_m_qznm1, Czn):
        lg = self.logmvnpdf_g(yn - Czn, self.params['sig_y'])
        lf = self.logmvnpdf_f(sv_A, ind0, ind1, indy, zn_m_qznm1, self.params["mcmc_N"])
        max_lf = np.max(lf)
        return (lg + max_lf - np.log(self.params["mcmc_N"]) + np.log(np.mean(np.exp(lf - max_lf))))

    def compute_pi_1(self, sv_A, ind0, ind1, indy, yn, zn_m_qznm1, Czn):
        lg = self.logmvnpdf_g(yn - Czn, self.params['sig_y'])
        lf = self.logmvnpdf_f(sv_A, ind0, ind1, indy, zn_m_qznm1, 1)
        return (lg + lf)

    def compute_pi_1_iidNoise(self, yn, zn_m_qznm1, Czn):
        lg = self.logmvnpdf_g(yn - Czn, self.params['sig_y'])
        lf = self.logmvnpdf_f_iidNoise(zn_m_qznm1, self.params['sig_x'])
        return (lg + lf)

    # -------------------------
    # helper to read observations for assimilation time corresponding to self.nstep
    # -------------------------
    def _get_obs_from_nc(self, nstep):
        """
        Return (y, obs_indices, indxob_level0) for assimilation at model step nstep.
        Mapping: saved obs index obs_t = (nstep+1)//assim_timesteps - 1
        (this matches your generation code where observations were written at nout-1)
        """
        # if obs file present, use loaded arrays
        if self.obs_file is not None and (self.yobs_all is not None or self.yobs_ind_all is not None):
            obs_t = (nstep + 1) // self.assim_timesteps - 1
            if (obs_t < 0) or (obs_t >= self.n_obs_times):
                return None, None, None
            y = None
            obs_indices = None
            indxob = None
            if self.yobs_all is not None:
                y = np.array(self.yobs_all[obs_t], copy=True)
            if self.yobs_ind_all is not None:
                obs_indices = np.array(self.yobs_ind_all[obs_t], dtype=int, copy=True)
            if self.yobs_ind_level0_all is not None:
                indxob = np.array(self.yobs_ind_level0_all[obs_t], dtype=int, copy=True)
            return y, obs_indices, indxob

        # fallback: use get_data/get_data_info as before (keeps behavior when no obs file provided)
        try:
            y = get_data(self.params, nstep, "data")
            obs_indices = get_data_info(self.params, nstep, 'obs_indices')
            if obs_indices is None:
                indxob = np.array([], dtype=int)
            else:
                # derive level-0 indices portion
                obs_indices = np.asarray(obs_indices, dtype=int)
                level0 = obs_indices[obs_indices < self.ncells]
                indxob = np.unique(level0)
            return y, obs_indices, indxob
        except Exception:
            return None, None, None

    # -------------------------
    # helper to get truth slice from nature-run NetCDF on demand
    # -------------------------
    def _get_truth_flat(self, truth_time_index):
        """
        Read pv[truth_time_index] from nature_run_file and return flattened array (dimx,)
        truth_time_index should correspond to the pv index in the saved NetCDF (pv[0] is initial state).
        """
        if self.nature_file is None:
            raise RuntimeError("nature_run_file not provided in params; cannot compute RMSE.")
        if not os.path.exists(self.nature_file):
            raise FileNotFoundError("nature_run_file '{}' not found".format(self.nature_file))

        with Dataset(self.nature_file, mode='r') as nc:
            if 'pv' not in nc.variables:
                raise RuntimeError("nature NetCDF does not contain variable 'pv'")
            # read single time slice
            pv_slice = np.array(nc.variables['pv'][truth_time_index], copy=True)
            flat = pv_slice.reshape(-1).astype(np.float32)
            if flat.size != self.dimx:
                raise RuntimeError("truth slice length mismatch {} != {}".format(flat.size, self.dimx))
            return flat

    # -------------------------
    # main run / sampling (MCMC inline, unchanged except for storage indexing)
    # -------------------------
    def run_main(self):
        print("Doing Local SMCMC Filtering (SQG) - Simul = %08d...." % self.params["isim"])
        starttime = time.time()

        print("h: %08d,   nstep = %05d" % (self.params["isim"], 0))

        # Ensure initial state for assimilation stored at index 0:
        if not np.any(self.lsmcmc_mean[0]):
            # try to populate initial state from params['x_star'] or nature file or get_data fallback
            if 'x_star' in self.params and self.params['x_star'] is not None:
                self.lsmcmc_mean[0] = np.asarray(self.params['x_star'], dtype=np.float32)
            elif self.nature_file is not None and os.path.exists(self.nature_file):
                try:
                    self.lsmcmc_mean[0] = self._get_truth_flat(0)
                except Exception:
                    self.lsmcmc_mean[0] = np.zeros((self.dimx,), dtype=np.float32)
            else:
                try:
                    # fallback to get_data at time index 0 (if available)
                    self.lsmcmc_mean[0] = get_data(self.params, 0, "signal")
                except Exception:
                    self.lsmcmc_mean[0] = np.zeros((self.dimx,), dtype=np.float32)

        

        for self.nstep in range(self.T):
            self.run_and_sample()

        print('Local SMCMC Filtering: Writing Simulation %d Results to File' % self.params["isim"])
        self._dump_results_nc()

        endtime = time.time() - starttime
        print('MCMC-Filtering: Simul = %d, dimx = %d, T = %d, Elapsed = %.3f'
              % (self.params["isim"], self.params["dimx"], self.params["T"], endtime))

    def _dump_results_nc(self):
        """
        Dump lsmcmc_mean (stored only at assimilation times) into a NetCDF.
        The time axis length will be nassim+1 (including initial state at index 0).
        Also save RMSE (length nassim).
        """
        outdir = self.params.get('lsmcmc_dir', '.')

        outfn = os.path.join(outdir, f"sqg_lsmcmc_out_{self.params['isim']:08d}.nc")
        with Dataset(outfn, mode='w', format='NETCDF4') as nc:
            # dimensions
            nc.createDimension('x', self.nx)
            nc.createDimension('y', self.ny)
            nc.createDimension('z', self.nlevs)
            nc.createDimension('t', self.nassim + 1)  # includes initial state (t=0)
            nc.createDimension('t_assim', self.nassim)  # assimilation times (no initial)
            # variables
            lsm = nc.createVariable('lsmcmc_mean', np.float32, ('t', 'z', 'y', 'x'), zlib=True)
            errvar = nc.createVariable('rmse_assim', np.float32, ('t_assim',), zlib=True)
            # reshape and write
            # lsmcmc_mean currently (dimx, nassim+1) -> reshape to (t,z,y,x)
            arr = self.lsmcmc_mean.reshape((self.nassim + 1, self.nlevs, self.ny, self.nx))
            lsm[:] = arr
            errvar[:] = self.RMSE

    # ---------- core step ----------
    def run_and_sample(self):
        # Advance the persistent forecast models by exactly one model timestep.
        # This ensures that after k calls to run_and_sample() the ensemble is at time k * dt.
        if self.nstep == 0:
            # initialize persistent forecast models from initial state
            self._init_forecast_models()
            self._advance_forecast_one_step()
        else:
            self._advance_forecast_one_step()
        # compute assimilation index when needed:
        # when (nstep+1) divisible by t_freq: assimilation occurs and assoc. assim_idx = (nstep+1)//t_freq
        # first assimilation when nstep = t_freq - 1 gives assim_idx = 1
        is_first_assim = ((self.nstep + 1) / self.params["t_freq"]) == 1
        is_later_assim = ((self.nstep + 1) % self.params["t_freq"] == 0) and (((self.nstep + 1) / self.params["t_freq"]) > 1)

        # ---------- first observation (exact sampling) ----------
        if is_first_assim:
            starttime = time.time()
            assim_idx = (self.nstep + 1) // self.params["t_freq"]  # == 1 here
        #     if self.forecast_models[0].t != self.obtimes[assim_idx]:
        #         raise ValueError('      Error: Mismatch between model ({}) and observation ({}) times'.\
        #                     format(self.forecast_models[0].t, self.obtimes[assim_idx]))

        #     # Use current ensemble mean as prior predictive mean (sol_flat).
        #     sol_flat = np.mean(self.forecast, axis=1)

        #     # get observations for this assimilation from obs NetCDF (preferred) or fallback
        #     y, obs_indices, self.indxob = self._get_obs_from_nc(self.nstep)
        #     if y is None or obs_indices is None:
        #         # fallback to previous behaviour (get_data)
        #         y = get_data(self.params, self.nstep, "data")
        #         obs_indices = get_data_info(self.params, self.nstep, 'obs_indices')
        #         if obs_indices is None:
        #             obs_indices = np.array([], dtype=int)
        #         # compute level0 observed indices array for get_observed_blocks_cells
        #         level0 = np.asarray(obs_indices, dtype=int)
        #         level0 = level0[level0 < self.ncells] if level0.size > 0 else np.array([], dtype=int)
        #         self.indxob = np.unique(level0)

        #     # get the observed-block expanded cell indices (full-state across levels)
        #     sv_ind_Q = self.get_observed_blocks_cells()

        #     noise = self.params['sig_x'] * np.random.normal(size=(self.dimx,))
        #     zn = sol_flat + noise

        #     # MCMC at observation time t1 (inline)
        #     old_pi = self.logmvnpdf_g(y - zn[obs_indices], self.params['sig_y']) + \
        #              self.logmvnpdf_f_iidNoise(noise[sv_ind_Q], self.params["sig_x"])

        #     count = 0
        #     zn_loc = zn[sv_ind_Q].copy()
        #     num_loc = len(sv_ind_Q)
        #     mcmc_samples = np.zeros((num_loc, self.params["mcmc_N"]))

        #     znp = np.copy(sol_flat)
        #     for i in range(self.mcmc_iters):
        #         prop_noise_loc = 1.0 * self.params["sig_mcmc_loc"] * np.random.normal(size=num_loc)
        #         znp[sv_ind_Q] = zn_loc + prop_noise_loc

        #         new_pi = self.logmvnpdf_g(y - znp[obs_indices], self.params['sig_y']) + \
        #                  self.logmvnpdf_f_iidNoise(znp[sv_ind_Q] - sol_flat[sv_ind_Q], self.params['sig_x'])

        #         alpha = new_pi - old_pi
        #         if np.log(np.random.uniform()) <= alpha:
        #             zn_loc = znp[sv_ind_Q].copy()
        #             old_pi = new_pi
        #             count += 1

        #         if i >= self.params["burn_in"]:
        #             mcmc_samples[:, i - self.params["burn_in"]] = zn_loc

        #     accep_rate = count / self.mcmc_iters

        #     if accep_rate > 0.25000:
        #         self.params["sig_mcmc_loc"] *= 1.15
        #     elif accep_rate < 0.20000:
        #         self.params["sig_mcmc_loc"] *= 0.85

        #     # build forecast by resampling posterior local samples and using zn as base
        #     Na = mcmc_samples.shape[1]
        #     Nf = self.params["nforecast"]
        #     dimx = self.dimx

        #     mcmc_samples = mcmc_samples.T  # (Na, num_loc)
        #     res_idx = np.random.choice(np.arange(Na), size=Nf, replace=True)
        #     analysis = np.empty((dimx, Nf), dtype=float)
        #     for k, idx in enumerate(res_idx):
        #         base = zn.copy()
        #         base[sv_ind_Q] = mcmc_samples[idx]
        #         analysis[:, k] = base


        #     # store analysis into lsmcmc_mean at assimilation index 1 (first)
        #     self.lsmcmc_mean[assim_idx] = np.mean(analysis, axis=1)
        #     self.lsmcmc_mean[assim_idx,sv_ind_Q] = np.mean(mcmc_samples, axis=0)

        #     # filter_mean_to_replicate = self.lsmcmc_mean[assim_idx].copy()
        #     # self.forecast[:,0:2] = np.tile(filter_mean_to_replicate[:, None], (1, 2)).astype(float)
        #     # compute RMSE against truth slice pv[assim_idx] (pv[0] is initial, pv[assim_idx] matches assimilation)
            try:
                truth_flat = self._get_truth_flat(assim_idx)
                ## eventually we will plot self.scalefact * lsmcmc_mean
                #so better include it in the computation of RMSE
                err_vec = self.scalefact**2 * (self.forecast.mean(axis=1) - truth_flat)**2
                self.RMSE[assim_idx - 1] = np.sqrt(err_vec.mean())
            except Exception as e:
                print("Warning: could not compute RMSE at assimilation {}: {}".format(assim_idx, e))

            self.lsmcmc_mean[assim_idx] = np.mean(self.forecast, axis=1)
            accep_rate =1
            print('isim %d: Accep_rate at assim %d = %0.4f, RMSE = %.4f, cpu time = %.3f' % (
                self.params["isim"], assim_idx, accep_rate, self.RMSE[assim_idx - 1], time.time() - starttime))

        # # ---------- later observation times ----------
        elif is_later_assim:
            starttime = time.time()

            assim_idx = (self.nstep + 1) // self.params["t_freq"]  # == 1 here
        #     if self.forecast_models[0].t != self.obtimes[assim_idx]:
        #         raise ValueError('      Error: Mismatch between model ({}) and observation ({}) times'.\
        #                     format(self.forecast_models[0].t, self.obtimes[assim_idx]))

        #     y, obs_indices, self.indxob = self._get_obs_from_nc(self.nstep)
        #     if y is None or obs_indices is None:
        #         y = get_data(self.params, self.nstep, "data")
        #         obs_indices = get_data_info(self.params, self.nstep, 'obs_indices')
        #         if obs_indices is None:
        #             obs_indices = np.array([], dtype=int)
        #         level0 = np.asarray(obs_indices, dtype=int)
        #         level0 = level0[level0 < self.ncells] if level0.size > 0 else np.array([], dtype=int)
        #         self.indxob = np.unique(level0)

        #     sv_ind_Q = self.get_observed_blocks_cells()
        #     j = np.random.choice(self.params["nforecast"])  # initialize at random particle

        #     # berv_Z_tn: use current ensemble snapshot (we maintain persistent models)
        #     berv_Z_tn = self.forecast.copy()

        #     noise = self.params['sig_x'] * np.random.normal(size=(self.dimx,))
        #     Z_tn = berv_Z_tn[:, j] + noise

        #     old_pi = self.compute_pi_1_iidNoise(y, noise[sv_ind_Q], Z_tn[obs_indices])
        #     old_j = j
        #     choices = np.array([0, 1, -1], dtype=int)
        #     q_prob = 0.33
        #     prob_forward = 1.1*q_prob
        #     prob_backward = q_prob
        #     prob_center = 1-(prob_forward+prob_backward)
        #     prob = np.array([prob_center, prob_forward, prob_backward])

        #     count = 0
        #     Z_tn_loc = Z_tn[sv_ind_Q].copy()
        #     num_loc = len(sv_ind_Q)
        #     Z_tn_p = np.copy(Z_tn)
        #     mcmc_samples = np.zeros((num_loc, self.params["mcmc_N"]))
        #     mcmc_js = np.zeros((self.params["mcmc_N"],), dtype=int)
        #     mcmc_logps = np.zeros((self.params["mcmc_N"],), dtype=float)  # thinned log-posteriors
        #     best_logp = -np.inf
        #     best_local = None
        #     best_j = None
        #     best_candidate = None

        #     for i in range(self.mcmc_iters):
        #         noise_loc = self.params["sig_mcmc_loc"] * np.random.normal(size=num_loc)
        #         Z_tn_p[sv_ind_Q] = Z_tn_loc + noise_loc
        #         step = int(np.random.choice(choices, p=prob))
        #         j = (old_j + step) % self.params["nforecast"]
                
        #         # small guard / prebuild map for O(1) lookups
        #         prob_map = {int(c): float(p) for c, p in zip(choices, prob)}
        #         # if old_j == 0:
        #         #     j += 1
        #         # if old_j == self.params["nforecast"] - 1:
        #         #     j -= 1

        #         q_new_given_old = prob_map[step]            # q(k | old)
        #         q_old_given_new = prob_map[-step]           # q(-k | new)

        #         # if reverse-probability is zero then the reverse move is impossible and acceptance=0
        #         if q_old_given_new <= 0 or q_new_given_old <= 0:
        #             log_hastings = -np.inf
        #         else:
        #             log_hastings = np.log(q_old_given_new) - np.log(q_new_given_old)

        #         new_pi = self.compute_pi_1_iidNoise(y, Z_tn_p[sv_ind_Q] - berv_Z_tn[sv_ind_Q, j], Z_tn_p[obs_indices])

        #         if old_j == 0: 
        #             new_pi += np.log(prob_backward)
        #         if old_j == self.params["nforecast"] - 1:
        #             new_pi += np.log(prob_forward)

        #         alpha = new_pi - old_pi
        #         if np.log(np.random.uniform()) <= alpha:
        #             Z_tn_loc = Z_tn_p[sv_ind_Q].copy()
        #             old_pi = new_pi
        #             old_j = j
        #             count += 1

        #         if i >= self.params["burn_in"]:
        #             mcmc_samples[:, i - self.params["burn_in"]] = Z_tn_loc
        #             mcmc_js[i - self.params["burn_in"]] = int(old_j)
        #             mcmc_logps[i - self.params["burn_in"]] = old_pi

        #             if old_pi > best_logp:
        #                 best_logp = old_pi
        #                 best_local = Z_tn_loc.copy()
        #                 best_j = old_j
        #                 # build best candidate full-state from berv_Z_tn column best_j and overwrite local
        #                 cand = berv_Z_tn[:, best_j].copy()
        #                 cand[sv_ind_Q] = best_local
        #                 best_candidate = cand

        #     accep_rate = count / self.mcmc_iters

        #     if accep_rate > 0.28000:
        #         self.params["sig_mcmc_loc"] *= 1.05
        #     elif accep_rate < 0.20000:
        #         self.params["sig_mcmc_loc"] *= 0.95

        #     # select MAP candidate (best_candidate) if available; otherwise fall back to ensemble mean
        #     map_idx = int(np.nanargmax(mcmc_logps))
        #     chosen_map_idx = map_idx
        #     chosen_local = mcmc_samples[:, map_idx].copy()
        #     best_j = int(mcmc_js[map_idx])
        #     candidate = berv_Z_tn[:, best_j].copy()
        #     candidate[sv_ind_Q] = chosen_local
        #     assim_idx = (self.nstep + 1) // self.params["t_freq"]
        #     self.lsmcmc_mean[assim_idx] = candidate.copy()
            
            # Nf = self.params["nforecast"]
            # filter_mean_to_replicate = self.lsmcmc_mean[assim_idx].copy()
            # self.forecast[:,0:2] = np.tile(filter_mean_to_replicate[:, None], (1, 2)).astype(float)

            # # compute filter mean
            # Na = mcmc_samples.shape[1]
            # Nf = self.params["nforecast"]
            # dimx = self.dimx
            # all_js = mcmc_js.copy()
            # mcmc_samples = mcmc_samples.T
            # res_idx = np.random.choice(np.arange(Na), size=Nf, replace=True)
            # analysis = np.empty((dimx, Nf), dtype=float)
            # for k, idx in enumerate(res_idx):
            #     j0 = int(all_js[idx])
            #     base = berv_Z_tn[:, j0].copy()
            #     base[sv_ind_Q] = mcmc_samples[idx]
            #     analysis[:, k] = base


            # # store analysis into lsmcmc_mean at assimilation index
            # assim_idx = (self.nstep + 1) // self.params["t_freq"]
            # self.lsmcmc_mean[assim_idx] = np.mean(analysis, axis=1)
            # self.lsmcmc_mean[assim_idx,sv_ind_Q] = np.mean(mcmc_samples, axis=0)
            # self.forecast = analysis
            # compute RMSE against truth slice pv[assim_idx]
            try:
                truth_flat = self._get_truth_flat(assim_idx)
                ## eventually we will plot self.scalefact * lsmcmc_mean
                #so better include it in the computation of RMSE
                #err_vec = self.scalefact**2 * (self.lsmcmc_mean[assim_idx] - truth_flat) ** 2
                err_vec = self.scalefact**2 * (self.forecast.mean(axis=1) - truth_flat) ** 2
                self.RMSE[assim_idx - 1] = np.sqrt(err_vec.mean())
            except Exception as e:
                print("Warning: could not compute RMSE at assimilation {}: {}".format(assim_idx, e))

            self.lsmcmc_mean[assim_idx] = np.mean(self.forecast, axis=1)
            map_idx = 1
            accep_rate =1
            print('isim %d: Accep_rate at assim %d = %0.4f, RMSE = %.4f, best_j = %d, cpu time = %.3f' % (
                self.params["isim"], assim_idx, accep_rate, self.RMSE[assim_idx - 1], map_idx, time.time() - starttime))

            # infl = 0.01
            # mean = self.forecast.mean(axis=1, keepdims=True)
            # self.forecast = mean + (1.0 + infl) * (self.forecast - mean)
        # ---------- no-observation steps: forward dynamics ----------
        else:
            # Nothing to store at non-assimilation steps (we only save at assimilation times).
            # Forecast has already been advanced at the top of this method.
            # Optionally preserve diagnostics here.
            pass

        # bookkeeping: time and step counters
        dt_model = float(self.params.get("dt", 0.0))
        self.tsteps[self.nstep] = dt_model
        self.t_simul += self.tsteps[self.nstep]
        self.params["time"] = self.t_simul


    # -------------------------
    # local selection of indices as in original
    # -------------------------
    def get_ind_min_max(self, obs_indices):
        sv_ind_Q = np.array([], dtype=int)
        for i in range(len(self.params["subdomains_ind"])):
            sel = np.where(self.params["partition"].reshape(-1) == self.params["subdomains_ind"][i])[0]
            intersection = np.intersect1d(sel, obs_indices)
            if len(intersection) > 0:
                sv_ind_Q = np.hstack((sv_ind_Q, sel))
        sv_ind_Q = np.sort(sv_ind_Q, kind='mergesort')
        return sv_ind_Q

    def get_observed_blocks_cells(self):
        """
        Determine all grid cells that belong to any block that contains at least one level-0 observation.
        Returns flattened indices for the full state (all levels): e.g.
            [cells_lvl0, cells_lvl1 + ncells, ...]
        """
        if getattr(self, "indxob", None) is None or self.indxob.size == 0:
            return np.array([], dtype=int)

        # Convert flat observation indices (level-0) to 2D indices (y, x)
        yobs, xobs = np.unravel_index(self.indxob, (self.ny, self.nx))

        # Find the block index for each observed cell
        obs_block_ids = self.partition_labels[yobs, xobs]

        # Unique blocks that contain at least one observation
        unique_obs_blocks = np.unique(obs_block_ids)

        # Mask of selected cells in level-0
        mask = np.isin(self.partition_labels, unique_obs_blocks)

        # All (row, col) pairs of selected mask cells
        selected_ij = np.argwhere(mask)
        if selected_ij.size == 0:
            return np.array([], dtype=int)

        selected_flat_lvl0 = np.ravel_multi_index((selected_ij[:, 0], selected_ij[:, 1]), (self.ny, self.nx))

        # Expand to full-state across levels
        all_idx = []
        for lev in range(self.nlevs):
            offset = lev * self.ncells
            all_idx.append(selected_flat_lvl0 + offset)
        selected_full = np.concatenate(all_idx).astype(int)

        selected_full.sort()

        return selected_full

    # -------------------------
    # noise generator (kept unchanged)
    # -------------------------
    def compute_noise(self, N_samples, const=1):
        if N_samples == 1:
            epsil = const * self.params["var_sqrt"] * np.random.normal(
                size=(self.params['noise_modes_num'], self.params['noise_modes_num']))
            Xi = self.params["B_siny"] @ epsil @ self.params["A_sinx"].T
            noise = Xi.reshape(-1)
        else:
            epsil = const * self.params["var_sqrt"] * np.random.normal(
                size=(N_samples, self.params['noise_modes_num'], self.params['noise_modes_num']))
            Xi = self.params["B_siny"] @ epsil @ self.params["A_sinx"].T
            Xi = Xi.reshape(Xi.shape[0], Xi.shape[1] * Xi.shape[2])
            noise = Xi.T
        return noise
