import jax
jax.config.update("jax_enable_x64", True)
from equinox import filter_jit as jit, filter_vmap as vmap
import jax.numpy as jnp
from jax.scipy.special import sph_harm_y # used for constructing spherical harmonic basis
import jaxopt
import numpy as np, os

@jit
def signalResponse_fast(ptheta_a, pphi_a, gwtheta_a, gwphi_a, pair_idx_a, pair_idx_b):
    """
    A function to get the PTA response matrix (npair by npix). Adapted from ENTERPRISE.

    Args:
        ptheta_a (jax.Array or np.ndarray): A 1d array containing the pulsar theta coordinates in radians.
        pphi_a (jax.Array or np.ndarray): A 1d array containing the pulsar phi coordinates in radians.
        gwtheta_a (jax.Array or np.ndarray): A 1d array containing the GW theta coordinates in radians.
        gwphi_a (jax.Array or np.ndarray): A 1d array containing the GW phi coordinates in radians.
        pair_idx_a (jax.Array or np.ndarray): A 1d array of length N_pair containing the indices of pulsar a.
        pair_idx_b (jax.Array or np.ndarray): A 1d array of length N_pair containing the indices of pulsar b.

    Returns:
        tuple: A three-element tuple with the first element being the response matrix and the last two
            being the responses to plus and cross polarized graviational waves, respectively.
    """
    gwphi, pphi = jnp.meshgrid(gwphi_a, pphi_a)
    gwtheta, ptheta = jnp.meshgrid(gwtheta_a, ptheta_a)
    p = jnp.array([jnp.cos(pphi) * jnp.sin(ptheta), jnp.sin(pphi) * jnp.sin(ptheta), jnp.cos(ptheta)])
    Fp, Fc = createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, p)
    R = Fp[pair_idx_a]*Fp[pair_idx_b] + Fc[pair_idx_a]*Fc[pair_idx_b]
    return R, Fp, Fc
    
def createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, p):
    """
    A function to get the plus and cross polarized response matrices. Adapted from ENTERPRISE.

    Args:
        pphi (ArrayLike): ArrayLike containing the pulsar phi coordinates.
        ptheta (ArrayLike): ArrayLike containing the pulsar theta coordinates.
        gwphi (ArrayLike): ArrayLike containing the GW phi coordinates.
        gwtheta (ArrayLike): ArrayLike containing the GW theta coordinates.
        p (jax.Array or np.ndarray): Array containing the pulsar position unit vectors.

    Returns:
        tuple: A two-element tuple with the first element being the plus-polarized response and second being the cross-polarized
            response.
    """
    Omega = jnp.array([-jnp.sin(gwtheta) * jnp.cos(gwphi), -jnp.sin(gwtheta) * jnp.sin(gwphi), -jnp.cos(gwtheta)])
    mhat = jnp.array([-jnp.sin(gwphi), jnp.cos(gwphi), jnp.zeros(gwphi.shape)])
    nhat = jnp.array([-jnp.cos(gwphi) * jnp.cos(gwtheta), -jnp.cos(gwtheta) * jnp.sin(gwphi), jnp.sin(gwtheta)])
    npixels = Omega.shape[2]
    c = jnp.sqrt(1.5) / jnp.sqrt(npixels)
    Fp = 0.5 * c * (jnp.sum(nhat * p, axis=0) ** 2 - jnp.sum(mhat * p, axis=0) ** 2) / (1 - jnp.sum(Omega * p, axis=0))
    Fc = c * jnp.sum(mhat * p, axis=0) * jnp.sum(nhat * p, axis=0) / (1 - jnp.sum(Omega * p, axis=0))
    return Fp, Fc

def spherical_response(Fp, Fc, gwtheta, gwphi, l_max):
    """A function to compute the spherical harmonic basis antenna response matrix R_{clm, ab}.

    This function computes the spherical harmonics basis antenna response 
    matrix R_{clm, ab} where ab represents the pulsar pair made of pulsars 
    a and b, and k represents the pixel index.
    NOTE: This function uses the GW propogation direction for gwtheta and gwphi
    rather than the source direction (i.e. this method uses the vector from the
    source to the observer)

    Returns:
        jax.Array: An array of shape (nclm, npairs) containing the antenna
            pattern response matrix.
    """
    FpFc = jnp.zeros((Fp.shape[0], 2*Fp.shape[1])).at[:,0::2].set(Fp).at[:,1::2].set(Fc)

    lvals = jnp.concatenate([jnp.repeat(jnp.arange(l_max+1), jnp.array([2*ll + 1 for ll in range(l_max+1)]))])
    mvals = jnp.concatenate([jnp.arange(-ll, ll+1) for ll in range(l_max+1)])
    
    ylm_maps = compute_ylm_maps(lvals, mvals, gwtheta, gwphi, l_max)

    return _spherical_response(FpFc, ylm_maps)

@jit
def _spherical_response(FpFc, ylm_maps):
    """
    A function to compute the PTA response matrix in the spherical harmonic basis. Adapted from ENTERPRISE.

    Args:
        FpFc (jax.Array or np.ndarray): The PTA response matrices interweaved by polarization into a single matrix which has
            size npair by 2*npix.
        ylm_maps (jax.Array or np.ndarray): The spherical harmonics evaluated on a HEALPix grid. Get this from _compute_ylm_maps.
            Should be size nclm by npix.
    Returns:
        jax.Array: The response matrix of size nclm by npair.
    """
    ylm_maps_both_polarizations = jnp.repeat(ylm_maps, 2).reshape(ylm_maps.shape[0], -1)

    hdcov_F = jnp.dot(FpFc * ylm_maps_both_polarizations[:,None], FpFc.T)

    def add_pulsar_term(cov):
        return cov + jnp.diag(jnp.diag(cov))

    basis = vmap(add_pulsar_term)(hdcov_F)
    return basis

def make_alm2clm_cache(l_max):
    """
    A function to get square-root basis masks needed as inputs to alm2clm.

    Args:
        l_max (int): The l_max parameter of the search.

    Returns:
        list: The spherical harmonic basis masks for use in alm2clm.
    """
    alm2clm_cache = []
    lvals = jnp.concatenate([jnp.repeat(jnp.arange(l_max+1), jnp.array([2*ll + 1 for ll in range(l_max+1)]), total_repeat_length=(l_max+1)**2)]) #total_repeat_length=81)])
    mvals = jnp.concatenate([jnp.arange(-ll, ll+1) for ll in range(l_max+1)])
    abs_mvals = jnp.abs(mvals)
    clm_mask = abs_mvals * (2 * l_max + 1 - abs_mvals) // 2 + lvals
    mvals_positive = mvals > 0
    mvals_negative = mvals < 0
    mvals_float_power = jnp.float_power(-1, mvals)
    for mask in (clm_mask, mvals_float_power, mvals_positive, mvals_negative):
        alm2clm_cache.append(mask)
    return alm2clm_cache
    
def make_blm2alm_cache(l_max):
    """
    A function to get square-root basis masks needed as inputs to blm2alm.

    Args:
        l_max (int): The l_max parameter of the search (not divided by 2 yet, i.e., not blmax).

    Returns:
        list: The square-root basis masks for use in blm2alm.
    """
    blm2alm_cache = []
    blmax = l_max // 2
    precomputed_CG_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'precomputed_clebschGordan')
    if not os.path.exists(precomputed_CG_directory):
        os.makedirs(precomputed_CG_directory, exist_ok=True)
    precomputed_CG_filename = os.path.join(precomputed_CG_directory, 'lmax'+str(l_max)+'.npz')
    if os.path.exists(precomputed_CG_filename):
        sqrt_basis_helper = np.load(precomputed_CG_filename)
        beta_vals = sqrt_basis_helper['beta_vals']
        blvals = sqrt_basis_helper['blvals']
        bmvals = sqrt_basis_helper['bmvals']
    else:
        print('No precomputed CG coefficients found for this l_max. Computing them from scratch...')
        from . import clebschGordan as CG
        sqrt_basis_helper = CG.clebschGordan(l_max = l_max)
        print('Done.')
        beta_vals = jnp.array(sqrt_basis_helper.beta_vals)
        blvals = jnp.array(sqrt_basis_helper.bl_idx)
        bmvals = jnp.array(sqrt_basis_helper.bm_idx)

        print('Saving computed CG coefficients to MAPS directory for next time.')
        np.savez_compressed(precomputed_CG_filename, beta_vals=beta_vals, blvals=blvals, bmvals=bmvals)
        print('Done saving.')
        
    abs_bmvals = jnp.abs(bmvals)

    blm_mask = abs_bmvals * (2*blmax+1 - abs_bmvals) // 2 + blvals
    blm_vals_float_power = jnp.float_power(-1, bmvals)
    bmvals_negative = bmvals < 0
    bmvals_zero = bmvals[bmvals >= 0] == 0
    for mask in (blm_mask, blm_vals_float_power, bmvals_negative, beta_vals, bmvals_zero):
        blm2alm_cache.append(mask)
    return blm2alm_cache

def alm2clm(alm, alm2clm_cache):
    """
    A function to compute clm given alm. Adapted from ENTERPRISE.

    Args:
        alm (ArrayLike): A 1d array of alm values to be converted into clm values.
        sph_cache (list): A list of masks used during the conversion. Get this from make_sph_cache.

    Returns:
        ArrayLike: A 1d array of clm values.
    """
    clm = alm[alm2clm_cache[0]]
    clms_with_positive_m = alm2clm_cache[1] * jnp.real(clm) * jnp.sqrt(2)
    clms_with_negative_m = alm2clm_cache[1] * jnp.imag(clm) * jnp.sqrt(2)
    clm = jnp.where(alm2clm_cache[2], clms_with_positive_m, clm)
    clm = jnp.where(alm2clm_cache[3], clms_with_negative_m, clm)
    clm = jnp.real(clm)
    return clm * jnp.sqrt(4*jnp.pi) / clm[0]

def blm2alm(blms, blm2alm_cache):
    """
    A function to compute a set of alms given blms.

    Args:
        blms (ArrayLike): A 1d array of blm values to be converted into alm values.
        sqrt_cache (list): A list of masks used during the conversion. Get this from make_sqrt_cache.

    Returns:
        ArrayLike: A 1d array of alm values.
    """
    blm_full = blms[blm2alm_cache[0]]
    blms_with_negative_m = blm2alm_cache[1]*jnp.conj(blm_full)
    blm_full = jnp.where(blm2alm_cache[2], blms_with_negative_m, blm_full)
    alm_vals = jnp.einsum('ijk,j,k', blm2alm_cache[3], blm_full, blm_full)
    return alm_vals

    return _spherical_response(FpFc, ylm_maps)
