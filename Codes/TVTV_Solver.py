
import math
import numpy as np
from numpy.fft import fft2, ifft2

import torch

from scipy.sparse import csr_matrix, csc_matrix
from scipy.optimize import root, minimize

from interpolation import pil_resize, cv2_resize, torch_resize, scikit_resize, generate_down_sampler, generate_up_sampler
from utils import psf2otf_numpy, psf2otf_torch


def ConjGradNumpy(Fx, b, x, maxiter=500, tol=1E-6):    
    # Calculate initial residuals
    n = x.size
    res = b - Fx(x)
    pres = np.copy(res)
    rsold = np.sum(res**2)
    # Main Loop
    for i in range(maxiter):
        Fx_pres = Fx(pres)
        alpha = rsold / np.sum(pres*Fx_pres)
        x = x + alpha*pres

        if i > 1 and i % 99 == 1:
            res = b - Fx(x)
        else:
            res = res - alpha*Fx_pres
        
        rsnew = np.sum(res**2)

        if np.sqrt(rsnew) < tol:
            break

        pres = res + (rsnew/rsold)*pres
        rsold = rsnew
    
    # print("ConjGrad in TVTV Solver took:", i+1, " iters.")
    return x


def Smooth_TVTV_Solver_NUMPY(b: np.ndarray, w: np.ndarray, beta: float, epsilon: float, skip: bool, A, AT, AAT, D, DT, training: bool, maxiter:int):
    """
    Solves the problem:

        minimize  sum_{i=0}^{n} c(Dx_i) + beta * c(Dx_i - Dw_i)
            x
        subject to  Ax = b

        or the problem (when Skip Connection):
        minimize  sum_{i=0}^{n} c(Dx_i) + beta * c(Dx_i + Dw_i)
            x
        subject to  Ax = b - Aw
    
    where b: m x 1, D: 2n x n,  A: m x n,  w: n x 1, beta > 0 and  m < n.
    """
    P,Q = b.shape
    M,N = w.shape
    n = M*N
    m = P*Q

    # Flatten the inputs: b and w
    b = b.flatten()
    w = w.flatten()

    # For Skip connection across the TV-TV Layer
    if skip:
        w = -1.0 * w
        beta = np.float32(1/beta)
        b = b + A(w)
        
    dw = D(w)

    # Algorithm Parameters
    rho = np.float32(1.0)
    tau_rho = np.float32(10)
    mu_rho = np.float32(2)
    eps_prim = np.float32(1E-4)
    eps_dual = np.float32(1E-4)

    # Initialization of local variables
    aux = np.zeros_like(b)
    lam = np.zeros_like(w)
    x = -w if skip else w
    z = -w if skip else w
    
    chabbonier = lambda t: np.sqrt(t**2 + epsilon**2)
    chabbonier_dt = lambda t: (t / np.sqrt(t**2 + epsilon**2))

    # Main Iterations
    for k in range(maxiter):
        #=================== x-update  =====================#
        # We solve the x-minimization using scipy.optim.minimize with L-BFGS-B method
        def func(x):
            dx = D(x)
            dx_minus_dw = dx-dw
            lagrange_term = x-z
            fval = np.sum(chabbonier(dx) + beta * chabbonier(dx_minus_dw)) + np.dot(lam, lagrange_term) + 0.5*rho*np.dot(lagrange_term, lagrange_term)
            fgrad = DT(chabbonier_dt(dx) + beta * chabbonier_dt(dx_minus_dw)) + lam + rho * lagrange_term
            return fval, fgrad
        
        # Initial guess of solution
        options = {'disp': None, 'maxcor': 5, 'maxfun': 200, 'ftol': 1E-3, 'gtol': 1E-3, 'maxiter': 10, 'iprint': -1}
        result = minimize(func, x, method='L-BFGS-B', jac=True, options=options)
        x = result.x
        #=================== z-update ======================#
        z_prev = np.copy(z)
        p = (rho * x + lam) / rho
        Ap_minus_b = A(p) - b
        if training:  # In this case, AAT is already inverse of AAT matrix
            aux = AAT(Ap_minus_b)
        else:
            aux = ConjGradNumpy(AAT, Ap_minus_b, aux, maxiter=500, tol=1E-6)   
        z = p - AT(aux)
        #====================lam-update ====================#
        primal_res = x - z
        dual_res = -rho * (z - z_prev)
        lam = lam + rho * primal_res

        # ========= Adjusting RHO parameter  ============#
        primal_res_norm = np.linalg.norm(primal_res)
        dual_res_norm = np.linalg.norm(dual_res)

        if primal_res_norm > tau_rho * dual_res_norm:
            rho = mu_rho * rho
        elif dual_res_norm > tau_rho * primal_res_norm:
            rho = rho / mu_rho

        #======= Check for Convergence ===========#
        if primal_res_norm < eps_prim and dual_res_norm < eps_dual:
            break
    
    # print("ADMM took {:d} iterations.".format(k+1))
    print("L2-Norm of Primal Residual = {:0.8f} and Dual Residual = {:0.8f}.".format(primal_res_norm, dual_res_norm))
    
    z = np.expand_dims(z.reshape(M,N), 0)
    return z


def TVTV_Solver_NUMPY(b: np.ndarray, w: np.ndarray, beta: float, skip: bool, A, AT, AAT, D, DT, DTD: np.ndarray, training: bool, maxiter:int):
    P,Q = b.shape
    M,N = w.shape
    n = M*N
    m = P*Q

    # Flatten the inputs: b and w
    b = b.flatten()
    w = w.flatten()

    # For Skip connection across the TV-TV-Solver Layer
    if skip:
        w = -1.0 * w
        beta = np.float32(1/beta)
        b = b + A(w)
        
    dw = D(w)

    # Initialize primal and dual variables
    x = np.copy(w)
    u = np.copy(dw)
    v = np.copy(w)
    lam = np.zeros_like(dw)
    mu = np.zeros_like(w)

    # Primal and Dual Residuals
    primal_res = np.zeros(3*n, dtype=np.float32)
    dual_res = np.zeros(3*n, dtype=np.float32)

    # Auxillary variable for Conjugate Gradient
    aux = np.zeros_like(b)

    # ADMM parameters
    rho = np.float32(1.0)
    tau_rho = np.float32(10)
    mu_rho = np.float32(2)
    tol_primal = np.float32(1E-3)
    tol_dual = np.float32(1E-3)

    # ADMM Iterations
    for k in range(maxiter):
        # =================== Solving for u ===================== #
        s = lam - rho * D(v)
        rho_dw = rho*dw
        dw_pos = (dw >= 0)

        # Components for which dw_i >= 0
        indices = dw_pos & (s < -rho_dw - beta - 1.0)
        u[indices] = (-beta - 1.0 - s[indices]) / rho

        indices = dw_pos & (-rho_dw - beta - 1.0 <= s) & (s <= -rho_dw + beta - 1.0)
        u[indices] = dw[indices]

        indices = dw_pos & (-rho_dw + beta - 1.0 < s) & (s < beta - 1.0)
        u[indices] = (beta - 1.0 - s[indices]) / rho

        indices = dw_pos & (beta - 1.0 <= s) & (s <= beta + 1.0)
        u[indices] = 0.0

        indices = dw_pos & (s > beta + 1.0)
        u[indices] = (beta + 1.0 - s[indices]) / rho

        # Components for which dw_i < 0
        not_dw_pos = np.invert(dw_pos)
        indices = not_dw_pos & (s < -beta-1.0)
        u[indices] = (-beta - 1.0 - s[indices]) / rho

        indices = not_dw_pos & (-beta - 1.0 <= s) & (s <= -beta + 1.0)
        u[indices] = 0.0

        indices = not_dw_pos & (-beta + 1.0 < s) & (s < -rho_dw - beta + 1.0)
        u[indices] = (-beta + 1.0 - s[indices]) / rho

        indices = not_dw_pos & (-rho_dw - beta + 1.0 <= s) & (s <= -rho_dw + beta + 1.0)
        u[indices] = dw[indices]

        indices = not_dw_pos & (s > -rho_dw + beta + 1.0)
        u[indices] = (beta + 1 - s[indices]) / rho

        #============= Solving for x   ====================#
        p = (rho * v - mu) / rho
        Ap_minus_b = A(p) - b

        if training:  # In this case, AAT is already inverse of AAT matrix
            aux = AAT(Ap_minus_b)
        else:
            aux = ConjGradNumpy(AAT, Ap_minus_b, aux, maxiter=500, tol=1E-6)
            
        x = p - AT(aux)
        #==================== Solving for v  ===================#
        v_prev = np.copy(v)
        rhs = DT(u + lam/rho) + (mu / rho) + x
        lhs = DTD + 1.0
        v = (np.real(ifft2((fft2(rhs.reshape(M,N)) / lhs)))).astype(np.float32).flatten()

        #=============== Update the dual variables ==============#
        primal_res[:2*n] = u - D(v)
        primal_res[2*n:] = x - v

        dual_res[:2*n] = -rho * D(v - v_prev)
        dual_res[2*n:] = -rho * (v - v_prev)

        lam = lam + rho * primal_res[:2*n]
        mu = mu + rho * primal_res[2*n:]

        # ========= Adjusting RHO parameter  ============#
        primal_res_norm = np.linalg.norm(primal_res)
        dual_res_norm = np.linalg.norm(dual_res)

        if primal_res_norm > tau_rho * dual_res_norm:
            rho = mu_rho * rho
        elif dual_res_norm > tau_rho * primal_res_norm:
            rho = rho / mu_rho

        # Check for the Convergence of ADMM
        if (primal_res_norm < tol_primal) and (dual_res_norm < tol_dual):
            # print("TV-TV solver converged in {} iterations.".format(k+1))
            break

    # if (k == MAX_ITER - 1):
    print("TV-TV-solver took {} iters with Primal Norm: {} and  Dual Norm: {}".format(k+1, primal_res_norm, dual_res_norm))

    x = np.expand_dims(x.reshape(M,N), 0)
    return x


def Batch_TVTV_Solver_NUMPY(lr_inputs, cnn_outputs, hh, hv, beta, interp_scale, interp_order, interp_method, skip, training, maxiter):
    # Bring the inputs to CPU and detach them for computation graph to be able to cast to Numpy
    lr_inputs = lr_inputs.cpu().numpy().astype(np.float32)  # This has dimension = (batch_size, 1, P, Q)
    cnn_outputs = cnn_outputs.cpu().numpy().astype(np.float32)  # This has dimension = (batch_size, 1, M, N)
    
    if hh is None:
        hh = np.array([-1, 1], dtype=np.float32).reshape(1,2)
        # hh = torch.tensor([-1, 1], dtype=torch.float32).reshape(1,2)
    if hv is None:
        hv = np.array([-1, 1], dtype=np.float32).reshape(2,1)
        # hv = torch.tensor([-1, 1], dtype=torch.float32).reshape(2,1)

    B, C, M, N = cnn_outputs.shape
    B, C, P, Q = lr_inputs.shape
    n = M*N
    m = P*Q
            
    if interp_method == 'PIL':
        Aop = lambda x: pil_resize(x.reshape(M,N), (1/interp_scale), order=interp_order).flatten()
        ATop = lambda x: (pil_resize(x.reshape(P,Q), interp_scale, order=interp_order)).flatten()
    elif interp_method == 'CV':
        Aop = lambda x: cv2_resize(x.reshape(M,N), (1/interp_scale), order=interp_order).flatten()
        ATop = lambda x: (cv2_resize(x.reshape(P,Q), interp_scale, order=interp_order)).flatten()
    elif interp_method == 'TORCH':
        Aop = lambda x: torch_resize(x.reshape(M,N), (1/interp_scale), order=interp_order).flatten()
        ATop = lambda x: (torch_resize(x.reshape(P,Q), interp_scale, order=interp_order)).flatten()
    else:
        raise Exception("Unsupported Interpolation method!")

    AATop = lambda x: Aop(ATop(x))

    # Sparsifying Linear Operator
    HH = psf2otf_numpy(hh, (M,N))
    HV = psf2otf_numpy(hv, (M,N))
    DTD = (np.square(np.abs(HH)) + np.square(np.abs(HV))).astype(np.float32)

    def Dop(x: np.ndarray):
        x = x.reshape(M,N)
        x_fft = np.fft.fft2(x)
        dxH = np.real(np.fft.ifft2(x_fft * HH)).astype(np.float32).flatten()
        dxV = np.real(np.fft.ifft2(x_fft * HV)).astype(np.float32).flatten()
        return np.concatenate((dxH,dxV), axis=0)

    def DTop(z: np.ndarray):
        dx = z[0:n].reshape(M, N)
        dy = z[n:2*n].reshape(M, N)
        dxT = np.real(np.fft.ifft2(np.fft.fft2(dx) * np.conj(HH))).astype(np.float32)
        dyT = np.real(np.fft.ifft2(np.fft.fft2(dy) * np.conj(HV))).astype(np.float32)
        return (dxT + dyT).flatten()

    outputs = np.zeros_like(cnn_outputs)
    for i in range(B):
        outputs[i] = TVTV_Solver_NUMPY(lr_inputs[i].squeeze(), cnn_outputs[i].squeeze(), beta, skip, Aop, ATop, AATop, Dop, DTop, DTD, training, maxiter)

    return torch.from_numpy(outputs)


def Batch_Smooth_TVTV_Solver_NUMPY(lr_inputs, cnn_outputs, hh, hv, beta, epsilon, interp_scale, interp_order, interp_method, skip, training, maxiter):
    # Bring the inputs to CPU and detach them for computation graph to be able to cast to Numpy
    lr_inputs = lr_inputs.cpu().numpy().astype(np.float32)  # This has dimension = (batch_size, 1, P, Q)
    cnn_outputs = cnn_outputs.cpu().numpy().astype(np.float32)  # This has dimension = (batch_size, 1, M, N)
    if hh is None:
        hh = np.array([-1, 1], dtype=np.float32).reshape(1,2)
        # hh = torch.tensor([-1, 1], dtype=torch.float32).reshape(1,2)
    if hv is None:
        hv = np.array([-1, 1], dtype=np.float32).reshape(2,1)
        # hv = torch.tensor([-1, 1], dtype=torch.float32).reshape(2,1)

    B, C, M, N = cnn_outputs.shape
    B, C, P, Q = lr_inputs.shape
    n = M*N
    m = P*Q 
    
    if interp_method == 'PIL':
        Aop = lambda x: pil_resize(x.reshape(M,N), (1/interp_scale), order=interp_order).flatten()
        ATop = lambda x: (pil_resize(x.reshape(P,Q), interp_scale, order=interp_order)).flatten()
    elif interp_method == 'CV':
        Aop = lambda x: cv2_resize(x.reshape(M,N), (1/interp_scale), order=interp_order).flatten()
        ATop = lambda x: (cv2_resize(x.reshape(P,Q), interp_scale, order=interp_order)).flatten()
    elif interp_method == 'TORCH':
        Aop = lambda x: torch_resize(x.reshape(M,N), (1/interp_scale), order=interp_order).flatten()
        ATop = lambda x: (torch_resize(x.reshape(P,Q), interp_scale, order=interp_order)).flatten()
    else:
        raise Exception("Unsupported Interpolation method!")

    AATop = lambda x: Aop(ATop(x))

    # Sparsifying Linear Operator
    HH = psf2otf_numpy(hh, (M,N))
    HV = psf2otf_numpy(hv, (M,N))

    def Dop(x: np.ndarray):
        x = x.reshape(M,N)
        x_fft = np.fft.fft2(x)
        dxH = np.real(np.fft.ifft2(x_fft * HH)).astype(np.float32).flatten()
        dxV = np.real(np.fft.ifft2(x_fft * HV)).astype(np.float32).flatten()
        return np.concatenate((dxH,dxV), axis=0)

    def DTop(z: np.ndarray):
        dx = z[0:n].reshape(M, N)
        dy = z[n:2*n].reshape(M, N)
        dxT = np.real(np.fft.ifft2(np.fft.fft2(dx) * np.conj(HH))).astype(np.float32)
        dyT = np.real(np.fft.ifft2(np.fft.fft2(dy) * np.conj(HV))).astype(np.float32)
        return (dxT + dyT).flatten()

    outputs = np.zeros_like(cnn_outputs)
    for i in range(B):
        outputs[i] = Smooth_TVTV_Solver_NUMPY(lr_inputs[i].squeeze(), cnn_outputs[i].squeeze(), beta, epsilon, skip, Aop, ATop, AATop, Dop, DTop, training, maxiter)

    return torch.from_numpy(outputs)

