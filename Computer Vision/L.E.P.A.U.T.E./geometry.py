import torch
from globals import mps_safe

def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    with mps_safe(v.device):
        return _skew_symmetric_impl(v)

def _skew_symmetric_impl(v: torch.Tensor) -> torch.Tensor:
    B = v.shape[0]
    zero = torch.zeros(B, device=v.device, dtype=v.dtype)
    return torch.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero
    ], dim=1).view(B, 3, 3)

def se3_exp_map(xi: torch.Tensor) -> torch.Tensor:
    with mps_safe(xi.device):
        return _se3_exp_map_impl(xi)

def _se3_log_map_impl(T: torch.Tensor) -> torch.Tensor:
    B = T.shape[0]
    R, t = T[:, :3, :3], T[:, :3, 3]
    
    trace_R = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = torch.clamp((trace_R - 1.0) / 2.0, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    
    xi = torch.zeros(B, 6, device=T.device, dtype=T.dtype)
    
    mask_small = (theta < 1e-4)
    mask_pi = (theta > torch.pi - 1e-3)
    mask_large = ~(mask_small | mask_pi)
    
    phi_raw = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1]
    ], dim=1)
    
    if mask_large.any():
        th = theta[mask_large].unsqueeze(-1)
        sin_th = torch.sin(th)
        
        phi_l = (th / (2.0 * sin_th)) * phi_raw[mask_large]
        xi[mask_large, 3:] = phi_l
        
        K = _skew_symmetric_impl(phi_l)
        I = torch.eye(3, device=T.device, dtype=T.dtype).unsqueeze(0).expand(mask_large.sum(), -1, -1)
        half_th = th / 2.0
        
        sin_half_th = torch.sin(half_th)
        coef = 1.0 - (th * torch.cos(half_th)) / (2.0 * sin_half_th)
        
        V_inv = I - 0.5 * K + coef * torch.bmm(K, K) / (th**2)
        xi[mask_large, :3] = torch.bmm(V_inv, t[mask_large].unsqueeze(-1)).squeeze(-1)
        
    if mask_small.any():
        xi[mask_small, 3:] = 0.5 * phi_raw[mask_small]
        K = _skew_symmetric_impl(xi[mask_small, 3:])
        I = torch.eye(3, device=T.device, dtype=T.dtype).unsqueeze(0).expand(mask_small.sum(), -1, -1)
        V_inv = I - 0.5 * K + (1.0/12.0) * torch.bmm(K, K)
        xi[mask_small, :3] = torch.bmm(V_inv, t[mask_small].unsqueeze(-1)).squeeze(-1)
        
    if mask_pi.any():
        th = theta[mask_pi].unsqueeze(-1)
        R_pi = R[mask_pi]
        t_pi = t[mask_pi]
        
        A = (R_pi + torch.eye(3, device=T.device, dtype=T.dtype).unsqueeze(0)) / 2.0
        diag_A = torch.diagonal(A, dim1=-2, dim2=-1)
        max_idx = torch.argmax(diag_A, dim=-1)
        
        B_pi = R_pi.shape[0]
        batch_idx = torch.arange(B_pi, device=T.device)
        v_max = torch.sqrt(torch.clamp(diag_A[batch_idx, max_idx], min=1e-10))
        
        col = A[batch_idx, :, max_idx] 
        v = col / v_max.unsqueeze(-1) 
        v = v / torch.norm(v, dim=-1, keepdim=True)
        
        phi_raw_pi = phi_raw[mask_pi]
        sign_mask = torch.sum(v * phi_raw_pi, dim=-1) < 0
        v[sign_mask] = -v[sign_mask]
        
        phi_pi = th * v
        xi[mask_pi, 3:] = phi_pi
        
        K = _skew_symmetric_impl(phi_pi)
        I = torch.eye(3, device=T.device, dtype=T.dtype).unsqueeze(0).expand(mask_pi.sum(), -1, -1)
        
        half_th = th / 2.0
        sin_half_th = torch.sin(half_th)
        coef = 1.0 - (th * torch.cos(half_th)) / (2.0 * sin_half_th)
        V_inv = I - 0.5 * K + coef * torch.bmm(K, K) / (th**2)
        xi[mask_pi, :3] = torch.bmm(V_inv, t_pi.unsqueeze(-1)).squeeze(-1)
        
    return xi

def se3_log_map(T: torch.Tensor) -> torch.Tensor:
    with mps_safe(T.device):
        return _se3_log_map_impl(T)

def _se3_exp_map_impl(xi: torch.Tensor) -> torch.Tensor:
    B = xi.shape[0]
    rho, phi = xi[:, :3], xi[:, 3:]
    theta_sq = torch.sum(phi**2, dim=1, keepdim=True)
    theta = torch.sqrt(torch.clamp(theta_sq, min=1e-10))
    
    T = torch.eye(4, device=xi.device, dtype=xi.dtype).unsqueeze(0).repeat(B, 1, 1)
    K = _skew_symmetric_impl(phi)
    K2 = torch.bmm(K, K)
    
    mask_large = (theta > 1e-4).squeeze(1)
    mask_small = ~mask_large
    
    if mask_large.any():
        th = theta[mask_large].unsqueeze(-1)
        A_coef = torch.sin(th) / th
        B_term = (1.0 - torch.cos(th)) / (th**2)
        C = (1.0 - A_coef) / (th**2)
        
        K_l, K2_l = K[mask_large], K2[mask_large]
        I = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0).expand(mask_large.sum(), -1, -1)
        
        T[mask_large, :3, :3] = I + A_coef * K_l + B_term * K2_l
        V = I + B_term * K_l + C * K2_l
        T[mask_large, :3, 3] = torch.bmm(V, rho[mask_large].unsqueeze(-1)).squeeze(-1)

    if mask_small.any():
        K_s, K2_s = K[mask_small], K2[mask_small]
        I = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0).expand(mask_small.sum(), -1, -1)
        
        T[mask_small, :3, :3] = I + K_s + 0.5 * K2_s
        V = I + 0.5 * K_s + (1.0/6.0) * K2_s
        T[mask_small, :3, 3] = torch.bmm(V, rho[mask_small].unsqueeze(-1)).squeeze(-1)
        
    return T

def compose_poses(T_global: torch.Tensor, T_rel: torch.Tensor) -> torch.Tensor:
    with mps_safe(T_global.device):
        return torch.bmm(T_global, T_rel)