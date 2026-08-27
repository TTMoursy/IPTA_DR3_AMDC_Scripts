import jax
jax.config.update("jax_enable_x64", True)
from jax.lax import dynamic_slice
import jax.numpy as jnp

import numpy as np

def cpu_cache(psrs, pta, lfcore):
    npsrs = len(psrs)
    pair_idx = np.array(np.triu_indices(npsrs,1)).T
    xi = np.arccos([np.dot(psrs[a].pos, psrs[b].pos) for (a,b) in pair_idx])
    def xNy(N, x, y):
        term1 = N.Nmat.solve(y, left_array=x)
        term2 = N.MNF(x).T @ N.MNMMNF(y)
        return term1 - term2
    
    FNdt = []
    TNdt = []
    TNT = []
    FNT = []
    FNF = []
    for i in range(npsrs):
        
        F = pta._signalcollections[i]['gw'].get_basis()
        N = pta._signalcollections[i].get_ndiag()
        T = pta._signalcollections[i].get_basis()
        dt = pta._signalcollections[i].get_detres()
        
        FNdt.append(xNy(N,F,dt))
        TNdt.append(xNy(N,T,dt))
        TNT.append(xNy(N,T,T))
        FNT.append(xNy(N,F,T))
        FNF.append(xNy(N,F,F))
        
    return npsrs, jnp.array(FNdt), jnp.array(TNdt), jnp.array(TNT), jnp.array(FNT), jnp.array(FNF), jnp.array(pair_idx), jnp.array(xi)

def cpu_prep(npsrs, pta, lfcore, N, frequencies, pars_list_input=None):
    gw_sig = [s for s in pta._signalcollections[0].signals if s.signal_id=='gw'][0]

    phi = []
    phiinv = []
    pars_list = pars_list_input.copy() if pars_list_input is not None else []
    for noise_draw in range(N):
        _phiinv = []
        idx = np.random.randint(lfcore.burn, lfcore.chain.shape[0])
        if len(pars_list) < N:
            pars = dict(zip(lfcore.params, lfcore.chain[idx]))
            pars_list.append(pars)
        else:
            pars = pars_list[noise_draw]
        for psr_idx in range(npsrs):
            __phiinv = pta._signalcollections[psr_idx].get_phiinv(pars)
            if __phiinv.ndim == 1:
                __phiinv = np.diag(__phiinv)
            _phiinv.append(__phiinv)
        phiinv.append(_phiinv)
        phi.append(gw_sig.get_phi(pars))

    return jnp.array(phi), jnp.array(phiinv), pars_list

@jax.jit
def gpu_nmpfpcos(FNdt, TNdt, TNT, FNT, FNF, phiinv,  pair_idx, xi, phi, frequencies):
    X, Z = jax.vmap(jax.vmap(compute_XZ, in_axes=(0,0,0,0,0,0)), in_axes=(None,None,None,None,None,0))(FNdt, TNdt, TNT, FNT, FNF, phiinv) # vmap over noise draws and pulsars
    # orf matrix and orf design matrix
    def hd(xi): # only valid for cross-correlations; could do jnp.where(xi==0, 1, .5 - stuff) if want a general HD function
        x = (1-jnp.cos(xi)) / 2
        return 0.5 - x * ( 0.25 - 1.5*jnp.log(x) )
    hd_of_xi = hd(xi)
    a,b = pair_idx[:,0], pair_idx[:,1]
    orf_matrix = jnp.ones((X.shape[1],X.shape[1])).at[a,b].set(hd_of_xi).at[b,a].set(hd_of_xi)
    orf_design_matrix = hd_of_xi

    a,b = pair_idx[:,0], pair_idx[:,1]
    N = jnp.arange(Z.shape[0])
    def ZphiZphihat_n(n): # ZphiZphihat for a single noise-draw n
        ZphiZphihat = jnp.zeros((Z.shape[1],Z.shape[1], Z.shape[2],Z.shape[2]))
        ZphiZphihat = ZphiZphihat.at[a,b].set(((phi[n]*Z[n,a]) @ Z[n,b]))
        ZphiZphihat = ZphiZphihat.at[b,a].set(((phi[n]*Z[n,b]) @ Z[n,a]))
        return ZphiZphihat
    ZphiZphihat = jax.vmap(ZphiZphihat_n)(N) # vmap over noise draws

    def Zphi_n(n): # phi*Z for a single noise-draw n
        return phi[n]*Z[n]
    Zphi = jax.vmap(Zphi_n)(N)

    npair = Z.shape[1]*(Z.shape[1]-1)//2
    PoP_idx = jnp.array(jnp.triu_indices(npair)).T
    
    PoP = jnp.zeros((len(PoP_idx),4),dtype=int)
    PoP = PoP.at[:,(0,1)].set(pair_idx[PoP_idx[:,0]])
    PoP = PoP.at[:,(2,3)].set(pair_idx[PoP_idx[:,1]])

    # It is also helpful to create some basic filters. (Kyle)
    psr_match = (PoP[:,(0,1)] == PoP[:,(2,3)]) # checks (a==c,b==d) (Kyle)
    psr_inv_match = (PoP[:,(0,1)] == PoP[:,(3,2)]) # checks (a==d,b==c) (Kyle)

    conditions = [psr_match[:,0] & psr_match[:,1],
                  psr_match[:,0] & ~psr_match[:,1],
                  ~psr_inv_match[:,0] & psr_inv_match[:,1],
                  ~psr_match[:,0] & psr_match[:,1]]
    
    p_idx1, p_idx2 = PoP_idx.T
    a_PoP,b_PoP,c_PoP,d_PoP = PoP.T

    # a single-frequency PCOS
    def pcos(X, Z, ZphiZphihat, orf_matrix, phi, Zphi, k):
        # select the 2x2 frequency patch -- seems to save memory
        Xphihat = dynamic_slice(X, (0,2*k), (X.shape[0],2))
        Zphihat = dynamic_slice(Z, (0,2*k,2*k), (Z.shape[0],2,2))
        rho, sig, norm = compute_rhok_sigk(a, b, Xphihat, Zphihat, Z, phi, Zphi, k)
        s_diag = jnp.diag(sig**2)
        # select the 2x2 frequency patch -- seems to save memory
        ZphiZphihat = dynamic_slice(ZphiZphihat, (0,0,2*k,2*k), (Z.shape[0],Z.shape[0],2,2))
        C = create_PFOS_pair_covariance(Zphihat, ZphiZphihat, orf_matrix, norm, conditions, p_idx1,p_idx2, a_PoP,b_PoP,c_PoP,d_PoP)
        S = woodbury_linear_solve(orf_design_matrix, C, rho, s_diag)
        return rho, sig, S, C
    
    pfpcos = jax.vmap(pcos, in_axes=(None,None,None,None,None,None,0)) # vmap over frequencies
    nmpfpcos = jax.vmap(pfpcos, in_axes=(0,0,0,None,0,0,None)) # vmap over noise draws
    rhok, sigk, Sk, Ck = nmpfpcos(X, Z, ZphiZphihat, orf_matrix, phi, Zphi, frequencies)
    return rhok, sigk, Sk[:,:,0,0], Ck

def compute_XZ(FNdt, TNdt, TNT, FNT, FNF, phiinv):
    sigma = phiinv + TNT
    X = FNdt - FNT @ jnp.linalg.solve(sigma, TNdt)
    Z = FNF - FNT @ jnp.linalg.solve(sigma, FNT.T)
    return X, Z

def compute_rhok_sigk(a, b, Xphihat, Zphihat, Z, phi, Zphi, k):
    Zaslice = dynamic_slice(Z, (0,0,2*k), (Z.shape[0],Z.shape[1],2))
    Zbslice = dynamic_slice(Zphi, (0,2*k,0), (Z.shape[0],2,Z.shape[2]))

    norms_abk = phi[2*k]/mpt(Zaslice[a], Zbslice[b])

    rho_abk =  jnp.sum(Xphihat[a] * Xphihat[b], axis=1) * norms_abk
    sig_abk =  jnp.sqrt(mpt(Zphihat[a], Zphihat[b]) * norms_abk**2)

    return rho_abk, sig_abk, norms_abk
    
def woodbury_linear_solve(X, C, r, s):
    X = X[:,None]
    r = r[:,None]

    A = s
    K = C - A
    In = jnp.eye(A.shape[0])

    cinv = woodbury_inverse(A,In,In,K)

    fisher = X.T @ cinv @ X
    dirty_map = X.T @ cinv @ r 

    cov = jnp.linalg.pinv(fisher)
        
    theta = cov.T @ dirty_map
    return theta

def woodbury_inverse(A, U, C, V):
    A_inv_diag = 1/jnp.diag(A) 
    B_inv = jnp.linalg.inv(C + (V * A_inv_diag) @ U)
    return jnp.diag(A_inv_diag) - (A_inv_diag.reshape(-1,1) * U @ B_inv @ V * A_inv_diag)

def create_PFOS_pair_covariance(Zphihat, ZphiZphihat, orf, norm_ab, conditions, p_idx1,p_idx2, a,b,c,d):

    # Define the three cases for the pair covariance (Kyle - the rest of the comments in this function)
    def case1(a,b,c,d):             #(ab,cd)
        # = gamma_{ac} gamma_{bd} tr([Z_d phi Z_b] phihat [Z_a phi Z_c] phihat) + 
        #   gamma_{ad} gamma_{bc} tr([Z_c phi Z_b] phihat [Z_a phi Z_d] phihat)
        a4 = orf[a,c]*orf[d,b] * mpt(ZphiZphihat[d,b], ZphiZphihat[a,c]) + \
             orf[a,d]*orf[c,b] * mpt(ZphiZphihat[c,b], ZphiZphihat[a,d])
        return a4
    
    def case2(a,b,c):               #(ab,ac)
        # = gamma_{bc}            tr([Z_c phi Z_b] phihat [Z_a] phihat) + 
        #   gamma_{ac} gamma_{ab} tr([Z_a phi Z_b] phihat [Z_a phi Z_c] phihat)
        a2 = orf[b,c] *          mpt(ZphiZphihat[c,b],Zphihat[a])
        a4 = orf[a,c]*orf[a,b] * mpt(ZphiZphihat[a,b],ZphiZphihat[a,c])
        return a2+a4

    def case3(a,b):                 #(ab,ab)
        # = tr(Z_b phihat Z_a phihat) + gamma_{ab}^2 tr([Z_a phi Z_b] phihat [Z_a phi Z_b] phihat)
        a0 =               mpt(Zphihat[b],Zphihat[a])
        a4 = orf[a,b]**2 * mpt(ZphiZphihat[b,a],ZphiZphihat[b,a])
        return a0+a4

    choices = [case3(a,b), case2(a,b,d), case2(b,a,d), case2(b,a,c)]
    
    C_m = jnp.select(conditions, choices, default=case1(a,b,c,d))
    C_m = jnp.zeros((norm_ab.shape[0],norm_ab.shape[0])).at[p_idx1,p_idx2].set(C_m).at[p_idx2,p_idx1].set(C_m)

    # Include the final sigmas
    C_m *= jnp.outer(norm_ab,norm_ab)

    return C_m

def mpt(A,B):
    return jnp.einsum('ijk,ikj->i',A,B)
