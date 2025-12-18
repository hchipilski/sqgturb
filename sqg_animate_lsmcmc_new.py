import matplotlib.pyplot as plt
import matplotlib.animation as animation
from netCDF4 import Dataset
import numpy as np

# --- Load Data ---



nlevs = 2
N = 96
hours = 1
nsimu = 26
data_type = 'grid' #or 'swaths'

nature_filename = './example_lsmcmc_data/sqg_lsmcmc_nature_N96_%dhrly.nc' %hours
obs_filename = './example_lsmcmc_data/sqg_lsmcmc_obs_N96_%dhrly.nc' %hours


sims = []
for i in range(nsimu):
    analysis_filename = f"./example_lsmcmc/sqg_lsmcmc_out_{i:08d}.nc"
    nc_a = Dataset(analysis_filename)
    sims.append(nc_a["lsmcmc_mean"][:])
print(len(sims), sims[0].shape)
pv_a = np.zeros_like(sims[0], dtype=np.float32)
for arr in sims:
    pv_a += arr
pv_a /= nsimu

#pv_a = sims[0]
nsteps = pv_a.shape[0]
print("nassim=", nsteps)
nc_n = Dataset(nature_filename)
pv_n = nc_n['pv'][:]
t_var = nc_n['t'][0:nsteps]
x = nc_n['x'][:]
y = nc_n['y'][:]
scalefact = nc_n.f * nc_n.theta0 / nc_n.g


dx, dy = x[1] - x[0], y[1] - y[0]
Lx, Ly = x.max() - x.min(), y.max() - y.min()

# --- Swath Parameters ---
nc_data = Dataset(obs_filename)
swath_width = nc_data.swath_width #observatios consists of two swaths separated by a gap
gap_width = nc_data.gap_width
#swath_freq = nc_data.swath_freq
swath_freq = 20



##RMSE
rmse = np.empty(nsteps, np.float32)

for i in range(nsteps):
    rmse[i] = np.sqrt(((scalefact*pv_a[i].reshape(nlevs*N*N) - scalefact*pv_n[i].reshape(nlevs*N*N))**2).mean())
    print(rmse[i])
# --- Plot Setup ---
fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
ax0, ax1, ax2, ax3 = axes.flatten()   # flatten 2D array into list


ax0.set_aspect('equal')
ax1.set_aspect('equal')
ax3.set_aspect('equal')

vmin, vmax, levplot = -25, 25, 1

im0 = ax0.imshow(scalefact * pv_n[0, levplot], cmap='jet',
                 origin='lower', vmin=vmin, vmax=vmax)
ax0.set_title('Nature PV')
ax0.axis('off')

im1 = ax1.imshow(scalefact * pv_a[0, levplot], cmap='jet',
                 origin='lower', vmin=vmin, vmax=vmax)
ax1.set_title('Analysis PV')
ax1.axis('off')

mindiff = np.min(scalefact * pv_n[0:nsteps, levplot] - scalefact * pv_a[:, levplot])
maxdiff = np.max(scalefact * pv_n[0:nsteps, levplot] - scalefact * pv_a[:, levplot])
print(mindiff, maxdiff)
diff_data = scalefact * pv_n[0, levplot] - scalefact * pv_a[0, levplot]
im2 = ax3.imshow(diff_data, cmap='jet', origin='lower', vmin=mindiff, vmax=maxdiff)
ax3.set_title('Difference: Nature - Analysis')
ax3.axis('off')
cbar = fig.colorbar(im2, ax=ax3, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Difference')

nstep = 0
line, = ax2.plot([], [], 'k-', lw=2)
ax2.set_xlim(0, nsteps)
ax2.set_title('Time = %d' % nstep)
ax2.set_xlabel('Time step')
ax2.set_ylabel('RMS Error')
if np.isfinite(np.max(rmse)):
    ax2.set_ylim(0, 1.005 * np.max(rmse))
else:
    ax2.set_ylim(0, 10)

if data_type == 'swaths':
    swath_lines = [ax1.plot([], [], 'k--', linewidth=1.5, alpha=0.7)[0] for _ in range(12)]

rmse_times, rmse_vals = [], []

def updatefig(nstep):
    global rmse_times, rmse_vals
    ax2.set_title('Time = %d' % (nstep + 1))
 
    if nstep == 0:  # reset for a fresh cycle
        rmse_times = []
        rmse_vals = []


    nature_data = scalefact * pv_n[nstep, levplot]
    analysis_data = scalefact * pv_a[nstep, levplot]
    diff_data = nature_data - analysis_data

    im0.set_data(nature_data)
    im1.set_data(analysis_data)
    im2.set_data(diff_data)

    rmse_times.append(nstep)
    rmse_vals.append(rmse[nstep])
    line.set_data(rmse_times, rmse_vals)

    if data_type == 'swaths':
        # Swath geometry
        angle_deg = 70 if nstep % 2 == 0 else 110
        angle_rad = np.deg2rad(angle_deg)
        normal = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
        swath_dir = np.array([normal[1], -normal[0]])

        # Use the same shift formula as generate_swath_observations:
        # center_x = (frame * (Lx / nassim)) % Lx
        center_x = (nstep * (Lx / swath_freq)) % Lx + x.min()
        center = np.array([center_x, 0.5 * (y.max() + y.min())])

        line1 = -0.5 * (2 * swath_width + gap_width)
        line2 = line1 + swath_width
        line3 = line2 + gap_width
        line4 = line3 + swath_width
        offsets = np.array([line1, line2, line3, line4])

        shifts = [-Lx, 0, Lx]
        line_len = 2 * max(Lx, Ly)

        line_idx = 0
        for shift_x in shifts:
            shifted_center = center + np.array([shift_x, 0.0])
            for offset in offsets:
                shift = offset * normal
                p0 = shifted_center - line_len * swath_dir + shift
                p1 = shifted_center + line_len * swath_dir + shift

                ix0 = (p0[0] - x.min()) / dx
                iy0 = (p0[1] - y.min()) / dy
                ix1 = (p1[0] - x.min()) / dx
                iy1 = (p1[1] - y.min()) / dy

                swath_lines[line_idx].set_data([ix0, ix1], [iy0, iy1])
                line_idx += 1

    if data_type == 'swaths':
        return im0, im1, im2, line, *swath_lines
    else:
        return im0, im1, im2, line

ani = animation.FuncAnimation(fig, updatefig, frames=nsteps, interval=200, blit=False, repeat=True)
plt.show()
