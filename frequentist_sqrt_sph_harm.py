import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, jaxopt

import glob, sys, pickle
import numpy as np, healpy as hp
from enterprise.pulsar import Pulsar
from enterprise.signals import parameter, utils, signal_base, selections, white_signals, gp_signals, gp_priors as gpp, gp_bases as gpb
from enterprise_extensions.blocks import dm_noise_block
from la_forge import core
from defiant import nmpfpcos_jax_v1 as nj
from maps.updateC import (
    blm2alm, 
    alm2clm,
    make_blm2alm_cache, 
    make_alm2clm_cache,
    spherical_response,
    signalResponse_fast
)

my_id = int(sys.argv[1])
dataset_dir = sorted(glob.glob('/home/tmoursy/dr3/mdc/*/'))[my_id - 1]
with open(dataset_dir+'psrs.pkl', 'rb') as fp:
    psrs = pickle.load(fp)

tmin = [p.toas.min() for p in psrs]
tmax = [p.toas.max() for p in psrs]
Tspan = np.max(tmax) - np.min(tmin)
efac = parameter.Constant(1.0) 
tnequad = parameter.Constant(-6)
selection = selections.Selection(selections.no_selection)
log10_A = parameter.Uniform(-20, -11)
gamma = parameter.Uniform(0, 7)
log10_A_gw = parameter.Uniform(-20, -11)('log10_A_gw')
gamma_gw = parameter.Uniform(0, 7)('gamma_gw')
mn = white_signals.MeasurementNoise(efac=efac, selection=selection)
eq = white_signals.TNEquadNoise(log10_tnequad=tnequad, selection=selection)
pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
cpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
fbreak = 10**(
    core.Core(
        corepath='/lustre/research/npol/tmoursy/dr3/mdc/'+dataset_dir.split('/')[-2]+'/break.core',
        burn=0.5).get_param_median('gw_log10_fb')
)
fbreak = int(fbreak*Tspan)
curn = gp_signals.FourierBasisGP(spectrum=cpl, components=fbreak, Tspan=Tspan, name='gw')
tm = gp_signals.MarginalizingTimingModel(use_svd=True)
dmgp = dm_noise_block(Tspan=Tspan)
if 'hard' in dataset_dir:
    s = tm + mn + eq + rn + curn + dmgp
else:
    s = tm + mn + eq + rn + curn
models = []
for p in psrs:
    models.append(s(p))
pta = signal_base.PTA(models)
outDir = '/lustre/scratch/tmoursy/dr3/mdc/'+dataset_dir.split('/')[-2]
saveDir = '/lustre/research/npol/tmoursy/dr3/mdc/'+dataset_dir.split('/')[-2]
lfcore = core.Core(corepath = saveDir+'/curn.core')

print(fbreak)
freq_to_search_up_to = np.min((fbreak, 12))
frequencies = jnp.arange(freq_to_search_up_to)
N_noise_draws_at_a_time = 2
N_total = 1000
l_max = 8
nside = 8
psrs_theta, psrs_phi = jnp.array([p.theta for p in psrs]), jnp.array([p.phi for p in psrs])
gwtheta, gwphi = hp.pix2ang(nside, np.arange(12*nside**2))
pair_idx = jnp.array(jnp.triu_indices(len(psrs),1)).T
_, Fp, Fc = signalResponse_fast(psrs_theta, psrs_phi, gwtheta, gwphi, pair_idx[:,0], pair_idx[:,1])
Gamma_lm = spherical_response(Fp, Fc, gwtheta, gwphi, l_max)[:, pair_idx[:,0], pair_idx[:,1]]

blm2alm_cache = make_blm2alm_cache(l_max)
alm2clm_cache = make_alm2clm_cache(l_max)
initial_blms = np.random.uniform(-1,1,(24,))

def residuals(params, rho, Lt, N_inv):
    b00 = jnp.ones(1)
    blm = jnp.concatenate( (b00, params[:4], params[4::2]+1j*params[5::2]) )
    blm = jnp.where(blm2alm_cache[4], jnp.real(blm), blm)
    alm = blm2alm(blm, blm2alm_cache)
    clm = alm2clm(alm, alm2clm_cache)
    orf = clm @ Gamma_lm 
    A2 = orf.T @ N_inv @ rho / (orf.T @ N_inv @ orf)
    r = rho - A2*orf
    return Lt @ r

LM = jaxopt.LevenbergMarquardt(residuals, materialize_jac=True, jit=True).run

def sqrt_basis(rho, C, Gamma_lm, intitial_blms, HD):
    b00 = jnp.ones(1)
    N_inv = jnp.linalg.inv(C)
    Lt = jnp.linalg.cholesky(N_inv).T
    opt_params, _ = LM(initial_blms, rho, Lt, N_inv)
    blm = jnp.concatenate( (b00, opt_params[:4], opt_params[4::2]+1j*opt_params[5::2]) )
    blm = jnp.where(blm2alm_cache[4], jnp.real(blm), blm)
    alm = blm2alm(blm, blm2alm_cache)
    clm = alm2clm(alm, alm2clm_cache)
    orf = clm @ Gamma_lm
    A2 = orf.T @ N_inv @ rho / (orf.T @ N_inv @ orf)
    anis_res = rho - A2*orf
    iso_res = rho - HD
    snr2 = - (anis_res @ N_inv @ anis_res) + (iso_res @ N_inv @ iso_res)
    return A2, clm, snr2

v_sqrt_basis = jax.vmap(jax.vmap(jax.jit(sqrt_basis), in_axes=(0,0,None,None,None)), in_axes=(0,0,None,None,None))

npsrs, FNdt, TNdt, TNT, FNT, FNF, pair_idx, xi = nj.cpu_cache(psrs, pta, lfcore)
xx = (1 - jnp.cos(xi)) / 2
HD = 1.5 * xx * jnp.log(xx) - xx / 4 + 0.5

A2s, clms, snr2s, Ss = [],[],[],[]
for _ in range(N_total // N_noise_draws_at_a_time):

    phi, phiinv, _ = nj.cpu_prep(npsrs, pta, lfcore, N_noise_draws_at_a_time, frequencies)
    rhok, sigk, Sk, Ck = jax.block_until_ready(nj.gpu_nmpfpcos(FNdt, TNdt, TNT, FNT, FNF, phiinv, pair_idx, xi, phi, frequencies))
    A2, clm, snr2 = jax.block_until_ready(v_sqrt_basis(rhok/Sk[...,None], Ck/Sk[...,None,None]**2, Gamma_lm, initial_blms, HD))
    del Ck
    A2s.append(A2)
    clms.append(clm)
    snr2s.append(snr2)
    Ss.append(Sk)

np.save(saveDir + '/A2.npy', A2s)
np.save(saveDir + '/clm.npy', clms)
np.save(saveDir + '/snr2.npy', snr2s)
np.save(saveDir + '/S.npy', Ss)
