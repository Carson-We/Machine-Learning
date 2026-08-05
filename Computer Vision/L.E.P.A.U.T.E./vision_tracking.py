import threading
import contextlib
from typing import Tuple, List

import cv2
import torch
import numpy as np
import torch.nn.functional as F

from globals import logger, mps_safe
from pipeline_and_config import LepauteConfig
from geometry import se3_exp_map, se3_log_map
from ultralytics import YOLO

class MonocularDirectTracker:
    """
    Monocular tracking system handling direct photometric alignment with dynamic scale adjustment 
    and an intelligent feature-based ORB fallback mechanism.
    """
    def __init__(self, config: LepauteConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.dynamic_scale_prior = getattr(config, "object_scales", {}).get("object", 1.0)
        self.last_keyframe_features = None
        self.last_keyframe_img = None
        self.intrinsic_matrix = np.array([
            [config.fx, 0.0, config.cx],
            [0.0, config.fy, config.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        with mps_safe(self.device):
            scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32, device=self.device) / 16.0
            scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32, device=self.device) / 16.0
            self.kx = scharr_x.view(1, 1, 3, 3)
            self.ky = scharr_y.view(1, 1, 3, 3)
        
        logger.info(f"[MonocularDirectTracker] PyTorch Accelerated Direct Tracker Online. Device: {self.device} | Levels={self.config.pyramid_levels}")
        
    def update_dynamic_scale(self, new_scale: float):
        """
        Callback interface to dynamically update the tracker's internal scale prior 
        using closed-loop state estimation, eliminating static monocular drift.
        """
        if new_scale > 0.001:
            self.dynamic_scale_prior = new_scale
            logger.info(f"[MonocularDirectTracker] Scale prior synchronized dynamically to: {new_scale:.4f}")
            
    def track_fallback_orb(self, current_img: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Executes robust feature tracking when direct tracking fails.
        Uses the dynamic_scale_prior to resolve translation scaling.
        """
        if self.last_keyframe_img is None:
            self.last_keyframe_img = current_img
            kp, des = self.orb.detectAndCompute(current_img, None)
            self.last_keyframe_features = (kp, des)
            return np.eye(4), True

        kp_cur, des_cur = self.orb.detectAndCompute(current_img, None)
        kp_ref, des_ref = self.last_keyframe_features
        
        if des_cur is None or des_ref is None:
            return np.eye(4), False
            
        matches = self.matcher.match(des_ref, des_cur)
        if len(matches) < 12:
            return np.eye(4), False
            
        pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts_cur = np.float32([kp_cur[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Compute Essential Matrix using camera parameters
        E, mask = cv2.findEssentialMat(
            pts_cur, pts_ref, 
            cameraMatrix=self.intrinsic_matrix, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=1.0
        )
        
        if E is None or E.shape != (3, 3):
            return np.eye(4), False
            
        _, R, t, mask_pose = cv2.recoverPose(E, pts_cur, pts_ref, cameraMatrix=self.intrinsic_matrix, mask=mask)
        
        # Apply the dynamically adjusted scale prior to translation vector instead of a hardcoded constant
        t_scaled = t.flatten() * self.dynamic_scale_prior
        
        # Build relative transformation matrix
        T_rel = np.eye(4)
        T_rel[0:3, 0:3] = R
        T_rel[0:3, 3] = t_scaled
        
        # Update keyframe structures
        self.last_keyframe_img = current_img
        self.last_keyframe_features = (kp_cur, des_cur)
        
        return T_rel, True

    def _build_pyramid(self, img_tensor: torch.Tensor) -> List[torch.Tensor]:
        pyr = [img_tensor]
        # CRITICAL FIX: Properly reference the injected configuration block
        for l in range(self.config.pyramid_levels - 1):
            down = F.interpolate(pyr[-1].unsqueeze(0).unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False)
            pyr.append(down.squeeze(0).squeeze(0))
        return pyr

    def track(self, img_a: np.ndarray, img_b: np.ndarray, scale_prior: float = 1.0) -> Tuple[np.ndarray, float]:
        """
        Executes dense direct photometric alignment using a coarse-to-fine spatial pyramid,
        accelerated via PyTorch tensors, with dynamic fallback to ORB PnP tracking if photometric score is low.
        """
        if img_a is None or img_b is None:
            return np.zeros(6, dtype=np.float32), 0.0

        gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY) if len(img_a.shape) == 3 else img_a
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY) if len(img_b.shape) == 3 else img_b
        
        xi = torch.zeros(6, dtype=torch.float32, device=self.device)
        tracking_successful = False
        final_score = 0.0
        residuals = None
        valid_mask = None

        with mps_safe(self.device):
            try:
                t_a = torch.from_numpy(gray_a).to(dtype=torch.float32, device=self.device)
                t_b = torch.from_numpy(gray_b).to(dtype=torch.float32, device=self.device)
                
                pyr_a = self._build_pyramid(t_a)
                pyr_b = self._build_pyramid(t_b)
                
                base_max_iter = getattr(self.config, 'gn_max_iter', 10)

                for lvl in reversed(range(self.config.pyramid_levels)):
                    img_lvl_a = pyr_a[lvl]
                    img_lvl_b = pyr_b[lvl]
                    h, w = img_lvl_a.shape
                    
                    scale_factor = 1.0 / (2.0 ** lvl)
                    fx_l = self.config.fx * scale_factor
                    fy_l = self.config.fy * scale_factor
                    cx_l = self.config.cx * scale_factor
                    cy_l = self.config.cy * scale_factor
                    
                    img_b_batch = img_lvl_b.unsqueeze(0).unsqueeze(0)
                    gx = F.conv2d(img_b_batch, self.kx, padding=1).squeeze(0).squeeze(0)
                    gy = F.conv2d(img_b_batch, self.ky, padding=1).squeeze(0).squeeze(0)
                    
                    v_coords, u_coords = torch.meshgrid(
                        torch.arange(h, device=self.device), 
                        torch.arange(w, device=self.device), 
                        indexing='ij'
                    )
                    
                    level_max_iter = max(4, base_max_iter - lvl * 2)

                    for iter_idx in range(level_max_iter):
                        tx, ty, tz = xi[0], xi[1], xi[2]
                        wx, wy, wz = xi[3], xi[4], xi[5]
                        
                        Z = torch.full_like(u_coords, max(scale_prior, 0.1), dtype=torch.float32)
                        X = (u_coords - cx_l) / fx_l * Z
                        Y = (v_coords - cy_l) / fy_l * Z
                        
                        X_prime = X + (wz * Y - wy * Z) + tx
                        Y_prime = Y + (-wz * X + wx * Z) + ty
                        Z_prime = Z + (wy * X - wx * Y) + tz
                        Z_prime = torch.clamp(Z_prime, min=1e-4)
                        
                        u_prime = (X_prime / Z_prime) * fx_l + cx_l
                        v_prime = (Y_prime / Z_prime) * fy_l + cy_l
                        
                        valid_mask = ((u_prime >= 0) & (u_prime < w - 1) & (v_prime >= 0) & (v_prime < h - 1)).float()
                            
                        u_norm = (u_prime / (w - 1)) * 2.0 - 1.0
                        v_norm = (v_prime / (h - 1)) * 2.0 - 1.0
                        
                        u_norm = torch.clamp(u_norm, min=-2.0, max=2.0)
                        v_norm = torch.clamp(v_norm, min=-2.0, max=2.0)
                        u_norm = torch.where(torch.isnan(u_norm) | torch.isinf(u_norm), torch.zeros_like(u_norm), u_norm)
                        v_norm = torch.where(torch.isnan(v_norm) | torch.isinf(v_norm), torch.zeros_like(v_norm), v_norm)
                        
                        grid = torch.stack((u_norm, v_norm), dim=-1).unsqueeze(0)
                        
                        warped_b = F.grid_sample(img_b_batch, grid, align_corners=True, mode='bilinear').squeeze(0).squeeze(0)
                        warped_gx = F.grid_sample(gx.unsqueeze(0).unsqueeze(0), grid, align_corners=True, mode='bilinear').squeeze(0).squeeze(0)
                        warped_gy = F.grid_sample(gy.unsqueeze(0).unsqueeze(0), grid, align_corners=True, mode='bilinear').squeeze(0).squeeze(0)
                        
                        residuals = warped_b - img_lvl_a
                        
                        inv_z = 1.0 / Z_prime
                        inv_z2 = inv_z * inv_z
                        
                        du_dX = fx_l * inv_z
                        du_dY = torch.zeros_like(inv_z)
                        du_dZ = -fx_l * X_prime * inv_z2
                        
                        dv_dX = torch.zeros_like(inv_z)
                        dv_dY = fy_l * inv_z
                        dv_dZ = -fy_l * Y_prime * inv_z2
                        
                        J_X = warped_gx * du_dX + warped_gy * dv_dX
                        J_Y = warped_gx * du_dY + warped_gy * dv_dY
                        J_Z = warped_gx * du_dZ + warped_gy * dv_dZ
                        
                        J = torch.zeros((h, w, 6), dtype=torch.float32, device=self.device)
                        J[..., 0] = J_X
                        J[..., 1] = J_Y
                        J[..., 2] = J_Z
                        J[..., 3] = -J_Y * Z_prime + J_Z * Y_prime
                        J[..., 4] =  J_X * Z_prime - J_Z * X_prime
                        J[..., 5] = -J_X * Y_prime + J_Y * X_prime
                        
                        J_masked = J * valid_mask.unsqueeze(-1)
                        r_masked = residuals * valid_mask
                        
                        J_flat = J_masked.view(-1, 6)
                        r_flat = r_masked.view(-1)
                        
                        H = torch.matmul(J_flat.T, J_flat)
                        b = -torch.matmul(J_flat.T, r_flat)
                        
                        H += 1e-4 * torch.eye(6, dtype=torch.float32, device=self.device)
                        
                        try:
                            delta_xi = torch.linalg.solve(H, b)
                        except torch.linalg.LinAlgError:
                            break
                            
                        delta_xi = torch.where(torch.isfinite(delta_xi), delta_xi, torch.zeros_like(delta_xi))
                        xi += delta_xi
                        
                        if torch.linalg.norm(delta_xi) < 1e-4:
                            break
                            
                if residuals is not None and valid_mask is not None:
                    valid_count = valid_mask.sum()
                    if valid_count.item() > 16:
                        masked_residuals = residuals * valid_mask
                        mean_res = (torch.abs(masked_residuals).sum() / valid_count).item()
                        final_score = float(1.0 / (1.0 + mean_res))
                    else:
                        final_score = 0.0
                else:
                    final_score = 0.0
                    
                if np.isfinite(final_score) and torch.norm(xi).item() > 0:
                    tracking_successful = True
                
                xi_np = xi.cpu().numpy()
                    
            except Exception as alignment_exception:
                logger.warning(f"[MonocularDirectTracker] PyTorch tracking aborted: {alignment_exception}. Escalating to fallback.")
                xi_np = np.zeros(6, dtype=np.float32)

        enable_orb = getattr(self.config, 'enable_orb_fallback', True)
        if (not tracking_successful or enable_orb) and final_score < 0.15:
            xi_fallback, fallback_score = self._execute_orb_pnp_fallback(img_a, img_b, scale_prior)
            if fallback_score > 0.05 or not tracking_successful:
                xi_np = xi_fallback
                final_score = fallback_score
                
        if not np.all(np.isfinite(xi_np)):
            xi_np = np.zeros(6, dtype=np.float32)

        return xi_np, float(np.clip(final_score, 0.0, 0.95))

    def _execute_orb_pnp_fallback(self, img_a: np.ndarray, img_b: np.ndarray, scale_prior: float) -> Tuple[np.ndarray, float]:
        xi_out = np.zeros(6, dtype=np.float32)
        
        def to_uint8(img: np.ndarray) -> np.ndarray:
            if img.dtype != np.uint8:
                return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return img
            
        u_a, u_b = to_uint8(img_a), to_uint8(img_b)
        
        detector = cv2.ORB_create(nfeatures=750, scaleFactor=1.2, nlevels=4)
        kp_a, des_a = detector.detectAndCompute(u_a, None)
        kp_b, des_b = detector.detectAndCompute(u_b, None)
        
        if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
            return xi_out, 0.0
            
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        all_matches = sorted(matcher.match(des_a, des_b), key=lambda x: x.distance)[:150]
        
        if len(all_matches) < 8:
            return xi_out, 0.0
        
        pts_a = np.float32([kp_a[m.queryIdx].pt for m in all_matches])
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in all_matches])
        
        # CRITICAL FIX: Utilize dynamically updated configuration bounds, preventing AttributeError.
        K = np.array([[self.config.fx, 0.0, self.config.cx], 
                      [0.0, self.config.fy, self.config.cy], 
                      [0.0, 0.0, 1.0]], dtype=np.float32)
                      
        pts_3d = np.array([[(pt[0]-self.config.cx)/self.config.fx*scale_prior, 
                            (pt[1]-self.config.cy)/self.config.fy*scale_prior, 
                            scale_prior] for pt in pts_a], dtype=np.float32)
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_b, K, distCoeffs=None,
            iterationsCount=100, reprojectionError=2.0, confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success and inliers is not None and len(inliers) >= 4:
            xi_out[:3] = tvec.flatten()
            xi_out[3:] = rvec.flatten()
            return xi_out, float(len(inliers) / len(all_matches))
            
        return xi_out, 0.0

class YOLOClassifier:
    """
    YOLO-based object classification and detection subsystem replacing SigLIP.
    Maintains strict compatibility with the pipeline's inference worker interface.
    """
    def __init__(self, config: LepauteConfig, model_name: str = "yolov8n.pt"):
        self.config = config
        self.device = torch.device(config.device)
        logger.info(f"[YOLOClassifier] Loading YOLO model ({model_name}) on device: {self.device}")
        
        with mps_safe(self.device):
            self.model = YOLO(model_name)
        self.labels = config.object_names

    def predict(self, img: np.ndarray) -> Tuple[str, float]:
        logger.debug(f"[YOLOClassifier] Running YOLO inference on image shape: {img.shape}")
        try:
            with mps_safe(self.device):
                results = self.model(img, verbose=False, device=self.device)
            
            if results and len(results) > 0:
                res = results[0]
                
                if hasattr(res, 'probs') and res.probs is not None:
                    probs = res.probs
                    best_idx = torch.argmax(probs.data).item()
                    conf = probs.data[best_idx].item()
                    cls_id = int(probs.top1) if hasattr(probs, 'top1') else best_idx
                    
                    model_names = self.model.names
                    detected_name = model_names.get(cls_id, "background")
                    return detected_name, conf
                    
                elif hasattr(res, 'boxes') and res.boxes is not None and len(res.boxes) > 0:
                    boxes = res.boxes
                    confidences = boxes.conf
                    best_idx = torch.argmax(confidences).item()
                    conf = confidences[best_idx].item()
                    cls_id = int(boxes.cls[best_idx].item())
                    
                    model_names = self.model.names
                    detected_name = model_names.get(cls_id, "background")
                    return detected_name, conf
                
            fallback_label = "background" if "background" in self.config.object_names else (self.config.object_names[0] if self.config.object_names else "unknown")
            return fallback_label, 0.0
            
        except Exception as e:
            logger.error(f"[YOLOClassifier] YOLO inference failed: {e}. Operating in safety fallback mode.")
            fallback_label = "background" if "background" in self.config.object_names else (self.config.object_names[0] if self.config.object_names else "unknown")
            return fallback_label, 0.0
        
class ManifoldKinematicForecaster:
    """
    Kinematic State Estimator operating on the SE(3) x R manifold.
    Tracks 3D pose and estimates metric scale explicitly in log-space to handle scale drift.
    """
    def __init__(self, process_noise_pose: float = 1e-3, process_noise_scale: float = 1e-4):
        self.lock = threading.Lock()
        
        # State tracking variables
        self.current_pose = np.eye(4)       # SE(3) matrix representing global pose
        self.twist_velocity = np.zeros(6)    # se(3) tangent space linear/angular velocity
        self.log_scale = 0.0                # Logarithmic scale: sigma = ln(scale). Initial scale = 1.0
        self.log_scale_velocity = 0.0       # Time derivative of log scale
        
        self.last_timestamp = None
        self.q_pose = process_noise_pose
        self.q_scale = process_noise_scale

    def predict(self, timestamp: float) -> Tuple[np.ndarray, float]:
        """
        Predicts the SE(3) pose and metric scale at a future timestamp utilizing 
        the kinematic velocity and log-scale derivative on the Lie manifold.
        """
        with self.lock:
            if self.last_timestamp is None:
                self.last_timestamp = timestamp
                return self.current_pose.copy(), float(np.exp(self.log_scale))
                
            dt = timestamp - self.last_timestamp
            if dt <= 0:
                return self.current_pose.copy(), float(np.exp(self.log_scale))
                
            twist_t = torch.from_numpy(self.twist_velocity * dt).unsqueeze(0).float()
            with torch.no_grad():
                delta_pose = se3_exp_map(twist_t).squeeze(0).cpu().numpy()
            predicted_pose = self.current_pose @ delta_pose
            
            predicted_log_scale = np.clip(self.log_scale + self.log_scale_velocity * dt, -5.0, 5.0)
            predicted_scale = float(np.exp(predicted_log_scale))
            
            return predicted_pose, predicted_scale

    def update_state(self, measured_pose: np.ndarray, delta_xi: np.ndarray, delta_scale: float, timestamp: float, weight: float = 0.7):
        """
        Updates the manifold kinematic state with measured pose, relative twist delta, and scale updates.
        Employs adaptive sliding window noise estimation to prevent long-term tracking drift on SE(3) x R.
        """
        with self.lock:
            if not hasattr(self, 'state_history'):
                self.state_history = []
                
            if self.last_timestamp is None:
                self.current_pose = measured_pose
                self.log_scale = float(np.clip(np.log(max(delta_scale, 1e-4)), -5.0, 5.0))
                self.last_timestamp = timestamp
                return

            dt = timestamp - self.last_timestamp
            if dt <= 0:
                dt = 1e-3
                
            self.state_history.append({'dt': dt, 'delta_xi': delta_xi})
            if len(self.state_history) > 10:
                self.state_history.pop(0)

            adaptive_weight = weight
            if len(self.state_history) >= 3:
                recent_vels = np.stack([h['delta_xi'] / max(h['dt'], 1e-4) for h in self.state_history])
                var_vel = np.var(recent_vels, axis=0)
                mean_var = float(np.mean(var_vel))
                adaptive_weight = float(np.clip(weight / (1.0 + mean_var), 0.1, 0.9))

            with torch.no_grad():
                delta_xi_t = torch.from_numpy(delta_xi).unsqueeze(0).float()
                refined_measurement = measured_pose @ se3_exp_map(delta_xi_t).squeeze(0).cpu().numpy()
                
                pose_error_matrix = np.linalg.inv(self.current_pose) @ refined_measurement
                pose_err_t = torch.from_numpy(pose_error_matrix).unsqueeze(0).float()
                error_twist = se3_log_map(pose_err_t).squeeze(0).cpu().numpy()
                
                if np.linalg.norm(error_twist) > 5.0:
                    self.current_pose = refined_measurement
                    self.twist_velocity = error_twist / dt
                else:
                    error_twist_t = torch.from_numpy(adaptive_weight * error_twist).unsqueeze(0).float()
                    self.current_pose = self.current_pose @ se3_exp_map(error_twist_t).squeeze(0).cpu().numpy()
                    self.twist_velocity = (1.0 - adaptive_weight) * self.twist_velocity + adaptive_weight * (error_twist / dt)
            
            measured_log_scale = self.log_scale + np.log(max(delta_scale, 1e-4))
            log_scale_error = float(np.clip(measured_log_scale - self.log_scale, -1.0, 1.0))
            
            self.log_scale = float(np.clip(self.log_scale + adaptive_weight * log_scale_error, -5.0, 5.0))
            new_scale_vel = log_scale_error / dt
            self.log_scale_velocity = (1.0 - adaptive_weight) * self.log_scale_velocity + adaptive_weight * new_scale_vel
            
            self.last_timestamp = timestamp
            
    def get_scale(self) -> float:
        """ Returns the current absolute metric scale factor """
        with self.lock:
            return float(np.exp(self.log_scale))