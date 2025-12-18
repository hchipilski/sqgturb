import matplotlib
matplotlib.use('qtagg')
from sqgturb import SQG, rfft2, irfft2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# run SQG turbulence simulation, optionally plotting results to screen and/or saving to
# netcdf file.

# model parameters.

#N = 512 # number of grid points in each direction (waves=N/2)
#dt = 90 # time step in seconds
#diff_efold = 1800. # time scale for hyperdiffusion at smallest resolved scale

#N = 192
#dt = 300
#diff_efold = 86400./8.
#
#N = 128
#dt = 600
#diff_efold = 86400./3.

# N = 96
# dt = 900
# diff_efold = 86400./2.

def generate_swath_observations(
    x, y, Lx, Ly, frame, nassim,
    swath_width=50_000.0,
    gap_width=20_000.0,
    fixed_nobs=None,
    cycles_to_cross=20,   # NEW: how many cycles to go left->right
    oscillate=False,      # NEW: if True use back-and-forth (triangular) motion
    jitter=0.0,           # NEW: small random jitter in meters to break ties (0.0 disables)
    angle_even=70,
    angle_odd=110
):
    """
    Generate two swaths of observations mimicking SWOT mission.

    New args:
      cycles_to_cross: number of cycles to move the swath center from left to right
      oscillate: if True perform back-and-forth motion (triangular wave) across domain
      jitter: add small random jitter (meters) to the center_x to avoid repeated identical selections
    """

    nx, ny = x.shape[1], x.shape[0]
    dx = Lx / nx
    dy = Ly / ny

    angle_deg = angle_even if frame % 2 == 0 else angle_odd
    angle_rad = np.deg2rad(angle_deg)
    normal = np.array([-np.sin(angle_rad), np.cos(angle_rad)])

    # Determine center_x using cycles_to_cross and optionally oscillate
    if cycles_to_cross <= 0:
        cycles_to_cross = max(1, nassim)

    # linear speed per cycle to cross domain in cycles_to_cross cycles
    step = Lx / float(cycles_to_cross)

    if oscillate:
        # triangular wave: goes 0 -> Lx in cycles_to_cross, then Lx -> 0 in next cycles_to_cross, repeat
        period = 2 * cycles_to_cross
        pos = frame % period
        if pos <= cycles_to_cross:
            frac = pos / float(cycles_to_cross)   # 0..1 left->right
        else:
            frac = (period - pos) / float(cycles_to_cross)  # 1..0 right->left
        center_x = frac * Lx
    else:
        # wrap-around linear motion: increases by 'step' each cycle with modulo Lx
        center_x = (frame * step) % Lx

    # apply small jitter if requested
    if jitter and jitter > 0.0:
        center_x = (center_x + np.random.default_rng(frame).uniform(-jitter, jitter)) % Lx

    center = np.array([center_x, Ly / 2.0])

    # Flatten grid into (N, 2) shape
    pos = np.stack([x.ravel(), y.ravel()], axis=1)

    # Consider positions shifted by -Lx, 0, +Lx to simulate periodic wrapping (keeps continuity)
    shifts = [-Lx, 0.0, Lx]
    xobs_all = []
    yobs_all = []

    offset = (gap_width + swath_width) / 2.0
    half_w = swath_width / 2.0

    for shift in shifts:
        pos_shifted = pos.copy()
        pos_shifted[:, 0] += shift
        vecs = pos_shifted - center
        # distance along the normal direction
        dists = vecs @ normal

        in_swath = ((np.abs(dists + offset) <= half_w) |
                    (np.abs(dists - offset) <= half_w))

        selected = pos_shifted[in_swath]
        if selected.size > 0:
            # Wrap x back into base domain
            selected[:, 0] = np.mod(selected[:, 0], Lx)
            xobs_all.append(selected[:, 0])
            yobs_all.append(selected[:, 1])

    if len(xobs_all) == 0:
        # fallback: no points found (very narrow swath). Return empty arrays
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float), 0

    # Combine and deduplicate
    xobs = np.concatenate(xobs_all)
    yobs = np.concatenate(yobs_all)

    # Map to nearest grid indices (this is where small shifts can be lost if step < dx)
    ix = np.round(xobs / dx).astype(int)
    iy = np.round(yobs / dy).astype(int)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    inds = np.ravel_multi_index((iy, ix), (ny, nx))
    obs_inds = np.unique(inds)

    # If user wants fixed_nobs, either subsample or add nearest points
    if fixed_nobs is None:
        fixed_nobs = len(obs_inds)
    elif len(obs_inds) > fixed_nobs:
        obs_inds = np.random.choice(obs_inds, fixed_nobs, replace=False)
    elif len(obs_inds) < fixed_nobs:
        missing = fixed_nobs - len(obs_inds)
        all_indices = np.arange(nx * ny)
        outside = np.setdiff1d(all_indices, obs_inds, assume_unique=True)

        # Use Euclidean distance to the current swath center to pick extra points
        outside_x = x.ravel()[outside]
        outside_y = y.ravel()[outside]
        outside_pos = np.vstack((outside_x, outside_y)).T

        # compute distance
        dist_to_center = np.linalg.norm(outside_pos - center, axis=1)
        closest_indices = outside[np.argsort(dist_to_center)[:missing]]
        obs_inds = np.concatenate((obs_inds, closest_indices))

    xob = x.ravel()[obs_inds]
    yob = y.ravel()[obs_inds]

    return obs_inds, xob, yob, fixed_nobs



#######
N = 96
dt = 900    #seconds
nlevs = 2
diff_efold = 86400./2.
hours = 1 #each inteval before assimilation (output interval length)


norder = 8 # order of hyperdiffusion
dealias = True # dealiased with 2/3 rule?

# Ekman damping coefficient r=dek*N**2/f, dek = ekman depth = sqrt(2.*Av/f))
# Av (turb viscosity) = 2.5 gives dek = sqrt(5/f) = 223
# for ocean Av is 1-5, land 5-50 (Lin and Pierrehumbert, 1988)
# corresponding to ekman depth of 141-316 m over ocean.
# spindown time of a barotropic vortex is tau = H/(f*dek), 10 days for
# H=10km, f=0.0001, dek=100m.
dek = 0 # applied only at surface if symmetric=False
nsq = 1.0e-4; f=1.0e-4; g = 9.8; theta0 = 300
H = 10.0e3 # lid height
r = dek*nsq/f
U = 20 #17.5 # jet speed
Lr = np.sqrt(nsq)*H/f # Rossby radius
L = 20.*Lr
# thermal relaxation time scale
tdiab = 10.0*86400 # in seconds
symmetric = True # (if False, asymmetric equilibrium jet with zero wind at sfc)
# parameter used to scale PV to temperature units.
scalefact = f*theta0/g

# create random noise
pv = np.random.normal(0,100.,size=(nlevs,N,N)).astype(np.float32)
# add isolated blob on lid
nexp = 20
x = np.arange(0,2.*np.pi,2.*np.pi/N); y = np.arange(0.,2.*np.pi,2.*np.pi/N)
x,y = np.meshgrid(x,y)
x = x.astype(np.float32); y = y.astype(np.float32)
pv[1] = pv[1]+2000.*(np.sin(x/2)**(2*nexp)*np.sin(y)**nexp)
# remove area mean from each level.
for k in range(nlevs):
    pv[k] = pv[k] - pv[k].mean()

# get OMP_NUM_THREADS (threads to use) from environment.
threads = int(os.getenv('OMP_NUM_THREADS','1'))

# single or double precision
precision='single' # pyfftw FFTs twice as fast as double

# initialize qg model instance
model = SQG(pv,nsq=nsq,f=f,U=U,H=H,r=r,tdiab=tdiab,dt=dt,
            diff_order=norder,diff_efold=diff_efold,
            dealias=dealias,symmetric=symmetric,threads=threads,
            precision=precision,tstart=0)


#  initialize figure.
outputinterval = hours *3600. # interval between frames in seconds
tmin = 100.*86400. - outputinterval # time to start saving data (in days * seconds_per_day)
tmax = 300.*86400. # time to stop (in days)
nsteps = int(tmax/outputinterval) # number of time steps to animate
nassim = nsteps #number of assimilation steps
# set number of timesteps to integrate for each call to model.advance
timesteps = int(outputinterval/model.dt)
print("timesteps = ", timesteps)
model.timesteps = timesteps
savedata = './example_lsmcmc_data/sqg_lsmcmc_nature_N%s_%shrly.nc' % (N, hours)  # save data plotted in a netcdf file.
saveobs = './example_lsmcmc_data/sqg_lsmcmc_obs_N%s_%shrly.nc' % (N, hours)
#savedata = None # don't save data
plot = False # animate data as model is running?


## Observations
swath_width = 2500_000.0 #observatios consists of two swaths separated by a gap
gap_width = 850_000.0
swath_freq = 4
sig_y = 1.0 # observation error standard deviation in K (orig=1)
sig_x = 1.e-1
# if levob=0, sfc temp obs used.  if 1, lid temp obs used. If [0,1] obs at both
# boundaries.
levob = [0,1]; levob = list(levob); levob.sort() 


if savedata is not None:
    from netCDF4 import Dataset
    nc = Dataset(savedata, mode='w', format='NETCDF4_CLASSIC')
    nc.r = model.r
    nc.f = model.f
    nc.U = model.U
    nc.L = model.L
    nc.H = model.H
    nc.g = g; nc.theta0 = theta0
    nc.nsq = model.nsq
    nc.tdiab = model.tdiab
    nc.dt = model.dt
    nc.diff_efold = model.diff_efold
    nc.diff_order = model.diff_order
    nc.symmetric = int(model.symmetric)
    nc.dealias = int(model.dealias)
    nc.timesteps = model.timesteps
    nc.sig_x = sig_x
    x = nc.createDimension('x',N)
    y = nc.createDimension('y',N)
    z = nc.createDimension('z',2)
    t = nc.createDimension('t',None)
    pvvar =\
    nc.createVariable('pv',np.float32,('t','z','y','x'),zlib=True)
    pvvar.units = 'K'
    # pv scaled by g/(f*theta0) so du/dz = d(pv)/dy
    xvar = nc.createVariable('x',np.float32,('x',))
    xvar.units = 'meters'
    yvar = nc.createVariable('y',np.float32,('y',))
    yvar.units = 'meters'
    zvar = nc.createVariable('z',np.float32,('z',))
    zvar.units = 'meters'
    tvar = nc.createVariable('t',np.float32,('t',))
    tvar.units = 'seconds'
    xvar[:] = np.arange(0,model.L,model.L/N)
    yvar[:] = np.arange(0,model.L,model.L/N)
    zvar[0] = 0; zvar[1] = model.H


xvar,yvar = np.meshgrid(xvar,yvar)

# nobs = None
# #get number of observations in each level
# nobs = generate_swath_observations(xvar, yvar, model.L, model.L, 0,
#                                                  nassim, swath_width, gap_width, nobs, cycles_to_cross=swath_freq)[3]
nobs = 576

if len(levob) == 2:
    n_observations = 2*nobs

nc_data = Dataset(saveobs, mode='w', format='NETCDF4_CLASSIC')
nc_data.sig_y = sig_y
nc_data.swath_width = swath_width
nc_data.gap_width = gap_width
nc_data.swath_freq = swath_freq
n_obs = nc_data.createDimension('n_obs',n_observations)
t = nc_data.createDimension('t',None)
yobs_all =nc_data.createVariable('yobs_all',np.float32,('t','n_obs'),zlib=True)
yobs_ind_all =nc_data.createVariable('yobs_ind_all',np.int32,('t','n_obs'),zlib=True)
n_obs0 = nc_data.createDimension('n_obs0',nobs)
indxob_level0_all = nc_data.createVariable('yobs_ind_level0_all',np.int32,('t','n_obs0'),zlib=True)
tvar_data = nc.createVariable('t',np.float32,('t',))
tvar_data.units = 'seconds'



pvob = np.empty((len(levob),nobs),np.float32) #2D an empt array of observations



levplot = 1; nout = 0 # levplot < 0 is vertical mean PV

if plot:
    fig = plt.figure(figsize=(14,8))
    fig.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95)
    vmin = scalefact*model.pvbar[levplot].min()
    vmax = scalefact*model.pvbar[levplot].max()
    if levplot < 0:
        vmin=0.8*vmin; vmax=0.8*vmax
    def initfig():
        global im1,im2
        ax1 = fig.add_subplot(121)
        ax1.axis('off')
        pv = irfft2(model.pvspec[levplot])  # spectral to grid
        im1 = ax1.imshow(scalefact*pv,cmap=plt.cm.jet,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
        ax2 = fig.add_subplot(122)
        ax2.axis('off')
        pvspec_mean = model.meantemp()
        pv = irfft2(pvspec_mean)  # mean pv
        im2 = ax2.imshow(scalefact*pv,cmap=plt.cm.jet,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
        return im1,im2,
    def updatefig(*args):
        global nout
        model.advance()
        t = model.t
        pv = irfft2(model.pvspec[levplot])
        hr = t/3600.
        spd = np.sqrt(model.u[levplot]**2+model.v[levplot]**2)
        print(hr,spd.max(),scalefact*pv.min(),scalefact*pv.max())
        im1.set_data(scalefact*pv)
        pvspec_mean = model.meantemp()
        pv = irfft2(pvspec_mean)  # mean pv
        im2.set_data(scalefact*pv)
        if savedata is not None and t >= tmin:
            print('saving data at t  = %g hours' % hr)
            pvvar[nout,:,:,:] = irfft2(model.pvspec)
            tvar[nout] = t
            nc.sync()
            if t >= tmax: nc.close()
            nout = nout + 1
        return im1,im2,

    # interval=0 means draw as fast as possible
    ani = animation.FuncAnimation(fig, updatefig, frames=nsteps, repeat=False,\
          init_func=initfig,interval=0,blit=True)
    plt.show()
else:
    t = 0.0
    #because I want to add noise at every dt, I will set 
    model.timesteps = 1
    while t < tmax:
        for step in range(timesteps): #should be 12
            model.advance()
            t = model.t
            pv = irfft2(model.pvspec)
            pv += np.random.normal(scale=sig_x, size=(nlevs, N, N)) #adding noise every dt
            model.pvspec = rfft2(pv)
        hr = t/3600.
        spd = np.sqrt(model.u[levplot]**2+model.v[levplot]**2)
        print(hr,spd.max(),scalefact*pv.min(),scalefact*pv.max())
        if savedata is not None and t >= tmin:
            if nout >= 1:
                # # get locations of observations and their indices (on level 0)
                # indxob, xob, yob, _ = generate_swath_observations(xvar, yvar, model.L, model.L,
                #                             nout , nassim, swath_width, gap_width, nobs, cycles_to_cross=swath_freq)

                mask = np.zeros((N,N),dtype=bool)
                nskip = int(N/np.sqrt(nobs))
                # if every other grid point observed, shift every other time step
                # so every grid point is observed in 2 cycle.
                if nskip == 2 and self.ntime%2:
                    mask[1:N:nskip,1:N:nskip] = True
                else:
                    mask[0:N:nskip,0:N:nskip] = True
                tmp = np.arange(0,N*N).reshape(N,N)
                indxob = tmp[mask.nonzero()].ravel()
                #assign observations
                pvob = np.zeros((len(levob), nobs), dtype=np.float32)
                for k, lev in enumerate(levob):
                    pvob[k] = pv[lev].ravel()[indxob]          # same order for each level
                    pvob[k] +=  np.random.normal(scale=sig_y, size=nobs) # add ob errors

                yn = pvob.ravel()                              # order: level0(all obs), level1(all obs), ...
                # Build obs_indices in exactly the same order:
                obs_indices_blocks = []
                for lev in levob:
                    offset = lev * (N * N)
                    obs_indices_blocks.append(indxob + offset)
                obs_indices = np.concatenate(obs_indices_blocks).astype(np.int32)
                # store indxob (level-0 indices) under the dedicated var too

                yobs_all[nout-1] = yn
                yobs_ind_all[nout-1] = obs_indices 
                indxob_level0_all[nout-1] = indxob
                tvar_data[nout-1] = t

                

            print('saving data at t = %g hours' % hr)
            pvvar[nout,:,:,:] = pv
            tvar[nout] = t
            nc.sync()
            if t >= tmax: nc.close()
            nout = nout + 1


