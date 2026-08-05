import os
import time
import logging
import threading
import queue
import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional

from geometry import (
    skew_symmetric, se3_exp_map, se3_log_map, compose_poses,
)
from vision_tracking import (
    MonocularDirectTracker, YOLOClassifier, ManifoldKinematicForecaster,
)
from models import (
    SE3ResidualRefiner, MonocularSE3Warping, SE3CrossAttentionBlock,
)
from globals import logger, mps_safe
from pipeline_and_config import (
    LepauteConfig, DisplayMode, PerformanceMode, EquivariantDataset, SequenceDataCollector, train_sequence_loop, load_data, CameraIOStream
)

import argparse
import sys
import signal
import contextlib
import multiprocessing as mp

logger = logging.getLogger("LEPAUTE.Pipeline")

class GracefulShutdownHandler:
    """
    Listens for OS-level termination signals to trigger safe pipeline teardown,
    preventing SQLite database corruption and orphaned background threads.
    """
    def __init__(self):
        self.shutdown_requested = False
        self.sigint_count = 0
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except ValueError:
            # Safely bypass if instantiated outside the main thread
            pass

    def _signal_handler(self, sig, frame):
        self.sigint_count += 1
        
        if self.sigint_count == 1:
            logger.info(f"[Shutdown] Received termination signal ({sig}). Initiating graceful shutdown sequence...")
            self.shutdown_requested = True
            # CRITICAL FIX: Explicitly raise KeyboardInterrupt to break the main thread out of Python loops
            raise KeyboardInterrupt
            
        elif self.sigint_count == 2:
            logger.warning("[Shutdown] Graceful shutdown already in progress. The process might be blocked inside a C++ extension (e.g., PyTorch Compilation). Press Ctrl+C again to force abort.")
            
        else:
            # CRITICAL FIX: If the user presses Ctrl+C 3+ times, the system is deadlocked in C++ and ignoring Python exceptions. 
            # We must use os._exit to immediately terminate the process natively and bypass all Python runtime locks.
            logger.critical("[Shutdown] Multiple termination signals received. Process is deadlocked. Hard aborting via os._exit(130).")
            os._exit(130)

class InferenceWorker:
    """
    Background Inference Process Manager.
    Delegates heavy YOLO classification and PyTorch SE(3) Refiner models into a fully 
    isolated operating system process to completely bypass the Python GIL, 
    unblocking the high-frequency dense direct tracking loop on the main thread.
    """
    def __init__(self, config, model_path: str):
        self.config = config
        self.model_path = model_path
        
        self.ctx = mp.get_context('spawn')
        self.job_queue = self.ctx.Queue(maxsize=5)
        self.result_queue = self.ctx.Queue()
        
        self.running_event = self.ctx.Event()
        self.running_event.set()
        self.heartbeat_value = self.ctx.Value('d', time.time())
        self.drop_count = 0 
        
        self.process = self.ctx.Process(
            target=self._run_process,
            args=(
                self.config, 
                self.model_path, 
                self.job_queue, 
                self.result_queue, 
                self.running_event, 
                self.heartbeat_value
            ),
            daemon=True
        )
        
        # Updated signature to track the extracted delta_scale parameter and the async tracker_xi_rel history
        self.history_buffer: Dict[int, Tuple[Tuple[str, float], np.ndarray, float, np.ndarray, float, np.ndarray]] = {}
        self.state_lock = threading.Lock()
        
    def start(self):
        self.process.start()

    def is_alive(self):
        return self.process.is_alive()

    def is_healthy(self, timeout_sec: float = 15.0) -> bool:
        if not self.is_alive():
            return False
        return (time.time() - self.heartbeat_value.value) < timeout_sec

    def enqueue_job(self, frame_id: int, img_ref: np.ndarray, img_cur: np.ndarray, tracker_xi_rel: np.ndarray):
        try:
            self.job_queue.put_nowait((frame_id, img_ref.copy(), img_cur.copy(), tracker_xi_rel.copy()))
        except queue.Full:
            self.drop_count += 1
            if self.drop_count % 10 == 0:
                logger.warning(f"[InferenceWorker] Job queue full. Silently dropped {self.drop_count} frames so far to maintain system real-time throughput.")

    def get_latest_resolved_state(self, current_time: float) -> Optional[Tuple[Tuple[str, float], np.ndarray, float, np.ndarray, np.ndarray]]:
        """
        Retrieves the most recent asynchronous inference resolution from the separate process,
        updating local history buffering without blocking the main tracker.
        """
        with self.state_lock:
            # Drain IPC result queue to sync state
            while not self.result_queue.empty():
                try:
                    res = self.result_queue.get_nowait()
                    frame_id, state = res
                    self.history_buffer[frame_id] = state
                except queue.Empty:
                    break
                    
            # Expire old states (index 4 is completion_time)
            current_keys = list(self.history_buffer.keys())
            for k in current_keys:
                if current_time - self.history_buffer[k][4] > 5.0:
                    del self.history_buffer[k]
                    
            if not self.history_buffer:
                return None
                
            latest_resolved_id = max(self.history_buffer.keys())
            state = self.history_buffer.pop(latest_resolved_id)
            
            # Clean up older obsolete states to prevent memory leaks
            obsolete_keys = [k for k in list(self.history_buffer.keys()) if k <= latest_resolved_id]
            for k in obsolete_keys:
                self.history_buffer.pop(k, None)
                
            # Returns: (obj_name, conf), refined_xi_rel, delta_scale, uncertainty, async_tracker_xi_rel
            return state[0], state[1], state[2], state[3], state[5]

    def stop(self):
            logger.info("[InferenceWorker] Stop command received. Halting background process...")
            self.running_event.clear()
            
            # Flush the job queue to release potential blocked locks
            while not self.job_queue.empty():
                try:
                    self.job_queue.get_nowait()
                except queue.Empty:
                    break
            
            logger.info("[InferenceWorker] Waiting for process to cleanly join (timeout=2.0s)...")
            self.process.join(timeout=2.0)
            
            if self.process.is_alive():
                logger.warning("[InferenceWorker] Process join timed out. Force terminating zombie process...")
                self.process.terminate()
                self.process.join(timeout=1.0)
                
            logger.info("[InferenceWorker] Process successfully terminated.")
        
    @staticmethod
    def _run_process(config, model_path, job_queue, result_queue, running_event, heartbeat_value):
        """
        Independent Process Entrypoint.
        Models are instantiated locally in this memory space to prevent serialization faults.
        """
        
        import os, time, queue, torch
        from vision_tracking import YOLOClassifier
        from models import SE3ResidualRefiner
        from globals import logger, mps_safe
        
        device = torch.device(config.device)
        
        try:
            classifier = YOLOClassifier(config=config, model_name="yolov8n.pt")
            refiner = SE3ResidualRefiner(config=config, feature_dim=256, max_resolution=64).to(device)
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                if 'model_state_dict' in checkpoint:
                    refiner.load_compiled_state_dict(checkpoint['model_state_dict'])
                else:
                    refiner.load_compiled_state_dict(checkpoint)
                logger.info(f"[InferenceWorker] Loaded trained SE(3) refiner weights from: {model_path}")
            else:
                logger.warning(f"[InferenceWorker] Model {model_path} not found, utilizing random init fallback.")
            
            refiner.eval()
        except Exception as e:
            logger.error(f"[InferenceWorker] CRITICAL: Initialization fault in worker process: {e}")
            return
            
        while running_event.is_set():
            heartbeat_value.value = time.time()
            
            try:
                task = job_queue.get(timeout=0.1)
                frame_id, img_ref, img_cur, tracker_xi_rel = task
                
                obj_name, conf = classifier.predict(img_cur)
                
                t_ref = torch.from_numpy(img_ref).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                t_cur = torch.from_numpy(img_cur).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                t_xi = torch.from_numpy(tracker_xi_rel).float().unsqueeze(0)
                
                with torch.no_grad():
                    with mps_safe(device):
                        t_ref = t_ref.to(device)
                        t_cur = t_cur.to(device)
                        t_xi = t_xi.to(device)
                        
                        delta_xi, delta_scale, unc_pose, unc_scale = refiner(t_ref, t_cur, t_xi)
                        
                        refined_xi_rel = delta_xi.squeeze(0).cpu().numpy()
                        delta_scale_val = delta_scale.squeeze(0).cpu().item()
                        uncertainty = unc_pose.squeeze(0).cpu().numpy()
                        
                completion_time = time.time()
                
                try:
                    # FIX: Relay the original historic tracker_xi_rel back to maintain SE(3) consistency 
                    result_queue.put_nowait((
                        frame_id, 
                        ((obj_name, conf), refined_xi_rel, delta_scale_val, uncertainty, completion_time, tracker_xi_rel)
                    ))
                except queue.Full:
                    pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[InferenceWorker] Inference Subsystem Crash/Exception captured: {e}")
                continue

def init_components(config, display_mode, save_json, mock):
    """Initializes and returns all core execution components and threads."""
    shutdown_handler = GracefulShutdownHandler()
    
    min_frame_time = 0.0
    if config.performance_mode == PerformanceMode.LOW:
        config.pyramid_levels = max(1, config.pyramid_levels - 1)
        config.gn_max_iter = max(4, config.gn_max_iter // 2)
        min_frame_time = 0.066  
    elif config.performance_mode == PerformanceMode.HIGH:
        config.pyramid_levels = config.pyramid_levels + 1
        config.gn_max_iter = int(config.gn_max_iter * 1.5)
        min_frame_time = 0.0    

    stream = CameraIOStream(config=config, mock=mock)
    tracker = MonocularDirectTracker(config=config)
    collector = SequenceDataCollector(config=config)
    if save_json: collector.start()
        
    forecaster = ManifoldKinematicForecaster()
    
    model_path = "./checkpoints/best_model.pth"
    worker = InferenceWorker(config=config, model_path=model_path)
    worker.start()
    
    return shutdown_handler, min_frame_time, stream, tracker, collector, forecaster, worker

def process_frame(
    config, current_id, current_stamp, prev_rgb, frame_rgb, 
    worker, tracker, forecaster, T_global, 
    current_obj_name, latest_conf, latest_unc
):
    """Processes a single frame boundary and returns updated trajectory state arrays."""
    logger.debug(f"[Pipeline] Frame ID {current_id} routed to main thread. Evaluating asynchronous inference state...")
    
    async_state = worker.get_latest_resolved_state(current_stamp)
    if async_state is not None:
        # Destructure the historically aligned tracker parameter securely passed from the Async queue
        (obj_name, conf), refined_xi_rel, delta_scale_val, unc, async_tracker_xi_rel = async_state
        current_obj_name = obj_name 
        latest_conf = conf
        latest_unc = unc
        has_async = True
        logger.debug(f"[Pipeline] Async state resolved for Frame ID {current_id}: Target='{obj_name}' (Confidence: {conf:.2f}).")
    else:
        has_async = False
        delta_scale_val = 1.0
        
    scale_prior = config.object_scales.get(current_obj_name, 1.0)
    logger.debug(f"[Pipeline] Dispatching Dense Direct Tracker for Frame ID {current_id} | Scale Prior: {scale_prior:.3f}m")
    
    tracker_xi_rel, track_score = tracker.track(prev_rgb, frame_rgb, scale_prior)
    logger.debug(f"[Pipeline] Direct Tracker completed. Alignment Score: {track_score:.4f}")
    
    if track_score > 0.1 and (current_id % 10 == 0 or track_score < 0.4):
        worker.enqueue_job(current_id, prev_rgb, frame_rgb, tracker_xi_rel)
    
    if has_async:
        # Strictly correct SE(3) composition of tracked prior and network residual delta
        # FIX: Align the composition by using the specific historic `async_tracker_xi_rel` from the frame that spawned it, 
        # avoiding catastrophic superposition errors onto the latest tracker frame.
        with mps_safe(config.device):
            T_tracker = se3_exp_map(torch.from_numpy(async_tracker_xi_rel).float().unsqueeze(0).to(config.device))
            T_delta = se3_exp_map(torch.from_numpy(refined_xi_rel).float().unsqueeze(0).to(config.device))
            T_fused = compose_poses(T_tracker, T_delta)
            best_rel_xi = se3_log_map(T_fused).squeeze(0).cpu().numpy()
            
        mode = "Refined"
        applied_scale = scale_prior * delta_scale_val
    elif track_score >= 0.1:
        best_rel_xi = tracker_xi_rel
        mode = "Tracker"
        applied_scale = scale_prior
    else:
        logger.warning(f"[Pipeline] Tracking alignment failed for Frame ID {current_id} (Score < 0.1). Engaging Manifold Kinematic Forecaster.")
        pred_pose, pred_scale = forecaster.predict(current_stamp)
        
        with mps_safe(config.device):
            T_curr_global = torch.from_numpy(pred_pose).float().to(config.device).unsqueeze(0)
            T_rel_mat = torch.inverse(T_global) @ T_curr_global
            t_log = se3_log_map(T_rel_mat)
            best_rel_xi = t_log.squeeze(0).cpu().numpy()
        
        mode = "Recovery"
        applied_scale = pred_scale
        
    logger.debug(f"[Pipeline] Fusing state (Fusion Mode: {mode}). Updating SE(3) global trajectory...")
        
    with mps_safe(config.device):
        t_rel_tensor = torch.from_numpy(best_rel_xi).float().unsqueeze(0).to(config.device)
        T_rel = se3_exp_map(t_rel_tensor)
        T_global_clone = T_global.clone()
        T_global = compose_poses(T_global_clone, T_rel)
        T_global_log = se3_log_map(T_global)
        
        xi_global = T_global_log.squeeze(0).cpu().numpy()
        T_global_cpu = T_global.squeeze(0).cpu().numpy()
        
    # Providing the true evaluated relative motion vector for manifold kinematics tracking
    forecaster.update_state(T_global_cpu, best_rel_xi, applied_scale, current_stamp, weight=0.5)
    
    return (
        T_global, current_obj_name, latest_conf, latest_unc, 
        best_rel_xi, applied_scale, track_score, mode, xi_global
    )

def teardown(worker, collector, stream, display_mode, save_json):
    """Executes safe pipeline termination and OS-level memory deallocation."""
    logger.info("[Teardown] === STARTING TEARDOWN SEQUENCE ===")
    logger.info("[Teardown] Halting InferenceWorker...")
    worker.stop()
    
    if save_json:
        logger.info("[Teardown] Halting SequenceDataCollector...")
        collector.stop()
        
    logger.info("[Teardown] Releasing CameraIOStream hardware locks...")
    stream.release()
    
    if display_mode in (DisplayMode.REALTIME, DisplayMode.DETAILEDGUI):
        logger.info("[Teardown] Destroying OpenCV windows...")
        cv2.destroyAllWindows()
        
    logger.info("[Teardown] === PIPELINE SUBSYSTEM TORN DOWN SECURELY ===")


def run_pipeline(
    config, 
    display_mode: DisplayMode = DisplayMode.HEADLESS,
    unlimited: bool = False,
    save_json: bool = True,
    mock: bool = False
) -> List[Dict[str, Any]]:
    
    logger.info(f"[Pipeline] Initializing Hardened Monocular LEPAUTE Subsystem under [{config.performance_mode.value.upper()}] profile.")
    
    (shutdown_handler, min_frame_time, stream, tracker, 
     collector, forecaster, worker) = init_components(config, display_mode, save_json, mock)
        
    success, prev_rgb, prev_meta = stream.read()
    if not success: 
        logger.error("[Pipeline] Camera acquisition failed permanently. Terminating.")
        return []

    processed_payloads = []
    max_frames = 50 if not unlimited else 999999
    
    with mps_safe(config.device):
        T_global = torch.eye(4, dtype=torch.float32, device=config.device).unsqueeze(0)
        
    prev_xi = np.zeros(6, dtype=np.float32)
    last_stamp = prev_meta["timestamp"]
    last_heartbeat_check = time.time()
    last_wall_time = time.time()
    
    current_obj_name = "background"
    trajectory_2d = []  
    latest_conf = 0.0
    latest_unc = np.zeros(6, dtype=np.float32)
    
    if display_mode in (DisplayMode.REALTIME, DisplayMode.DETAILEDGUI):
        cv2.namedWindow("LEPAUTE SE(3) Subsystem", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("LEPAUTE SE(3) Subsystem", 960, 720) 
    
    try:
        for frame_idx in range(max_frames):
            if shutdown_handler.shutdown_requested:
                logger.info("[Pipeline] Main loop broken by graceful shutdown protocol.")
                break
                
            current_time = time.time()
            if current_time - last_heartbeat_check > 5.0:
                if not worker.is_healthy():
                    logger.error("[Pipeline] SYSTEM HALT: InferenceWorker failed health check. Preventing zombie lock.")
                    break
                last_heartbeat_check = current_time

            loop_start_wall = time.time()
            success, frame_rgb, metadata = stream.read()
            if not success: continue

            current_stamp = metadata["timestamp"]
            current_id = metadata["frame_id"]
            dt = current_stamp - last_stamp
            
            (T_global, current_obj_name, latest_conf, latest_unc, 
             best_rel_xi, applied_scale, track_score, mode, xi_global) = process_frame(
                config, current_id, current_stamp, prev_rgb, frame_rgb, 
                worker, tracker, forecaster, T_global, 
                current_obj_name, latest_conf, latest_unc
            )
            
            with mps_safe(config.device):
                euclidean_trans = T_global[0, :3, 3].cpu().numpy()
                
            trajectory_2d.append((euclidean_trans[0], euclidean_trans[2]))
            
            if len(trajectory_2d) > 2000:
                trajectory_2d.pop(0)
            
            summary = {
                "frame_id": current_id,
                "category": current_obj_name, 
                "xi": xi_global.tolist(),
                "tracking_score": track_score, 
                "fusion_mode": mode
            }
            processed_payloads.append(summary)
            
            if save_json:
                collector.append_transition(prev_rgb, frame_rgb, best_rel_xi, current_obj_name)
                
            if display_mode == DisplayMode.JSON:
                print(f"[METRIC] Frame={summary['frame_id']:03d} | Perf={config.performance_mode.value.upper()} | Target={summary['category']:<12} | Mode={summary['fusion_mode']:<10} | Score={track_score:.2f}")
            
            elif display_mode in (DisplayMode.REALTIME, DisplayMode.DETAILEDGUI):
                gui_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                gui_frame_large = cv2.resize(gui_frame, (800, 600), interpolation=cv2.INTER_LINEAR)
                h_large, w_large = gui_frame_large.shape[:2] 
                
                fps = 1.0 / dt if dt > 0 else 0.0
                current_wall_time = time.time()
                real_dt = current_wall_time - last_wall_time
                last_wall_time = current_wall_time
                actual_fps = 1.0 / real_dt if real_dt > 0 else 0.0
                
                cv2.putText(gui_frame_large, f"Frame ID: {current_id:03d} | FPS: {fps:.1f} | Actual FPS: {actual_fps:.1f} | Mode: {mode}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if display_mode == DisplayMode.DETAILEDGUI:
                    cv2.putText(gui_frame_large, f"Obj: {current_obj_name} (Conf: {latest_conf:.2f})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                    cv2.putText(gui_frame_large, f"Active Scale: {applied_scale:.3f}m", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                    cv2.putText(gui_frame_large, f"Tracking Score: {track_score:.2f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if track_score > 0.1 else (0, 0, 255), 2)
                    
                    map_size = 150
                    map_img = np.zeros((map_size, map_size, 3), dtype=np.uint8)
                    cv2.rectangle(map_img, (0, 0), (map_size-1, map_size-1), (255, 255, 255), 1)
                    
                    if len(trajectory_2d) > 1:
                        pts = np.array(trajectory_2d, dtype=np.float32)
                        pt_min, pt_max = pts.min(axis=0), pts.max(axis=0)
                        rng = np.maximum(pt_max - pt_min, 1e-4)
                        norm_pts = ((pts - pt_min) / rng * (map_size - 30) + 15).astype(np.int32)
                        
                        for i in range(1, len(norm_pts)):
                            cv2.line(map_img, tuple(norm_pts[i-1]), tuple(norm_pts[i]), (0, 255, 0), 1)
                        cv2.circle(map_img, tuple(norm_pts[-1]), 5, (0, 0, 255), -1)
                        
                    if h_large >= map_size + 20 and w_large >= map_size + 20:
                        gui_frame_large[15:15+map_size, w_large-map_size-15:w_large-15] = map_img
                
                cv2.imshow("LEPAUTE SE(3) Subsystem", gui_frame_large)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("[Pipeline] Pipeline terminated by user inside GUI window.")
                    break
                    
            prev_rgb = frame_rgb
            prev_xi = best_rel_xi
            last_stamp = current_stamp
            
            if min_frame_time > 0.0:
                loop_elapsed = time.time() - loop_start_wall
                if loop_elapsed < min_frame_time:
                    time.sleep(min_frame_time - loop_elapsed)
                    
    except KeyboardInterrupt:
        logger.warning("[Pipeline] Pipeline Interrupted by Host (KeyboardInterrupt caught natively in main loop).")
    except Exception as e:
        logger.error(f"[Pipeline] Pipeline crashed due to unexpected runtime fault: {e}")
    finally:
        teardown(worker, collector, stream, display_mode, save_json)
        
    return processed_payloads

def validate_display_mode(value: str) -> DisplayMode:
    mapping = {
        "headless": DisplayMode.HEADLESS,
        "gui": DisplayMode.REALTIME,
        "realtime": DisplayMode.REALTIME,
        "json": DisplayMode.JSON,
        "detailedgui": DisplayMode.DETAILEDGUI
    }
    val = value.lower()
    if val in mapping:
        return mapping[val]
    raise argparse.ArgumentTypeError(f"Invalid mode: '{value}'. Allowed values are: {list(mapping.keys())}")

def validate_performance_mode(value: str) -> PerformanceMode:
    try:
        return PerformanceMode(value.lower())
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid perf profile: '{value}'. Allowed values are: {[e.value for e in PerformanceMode]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LEPAUTE SE(3) Monocular Subsystem Execution Pipeline Control"
    )
    
    parser.add_argument(
        "--mode",
        type=validate_display_mode,
        default=DisplayMode.REALTIME,
        help="Select running mode: headless, gui (real-time window mode), json (data output mode), detailedgui (HUD + Trajectory map)"
    )
    
    # New Performance profile flag argument
    parser.add_argument(
        "--perf",
        type=validate_performance_mode,
        default=PerformanceMode.MEDIUM,
        choices=["low", "medium", "high"],
        help="Select performance profile: low (slower update frequency/conserves resource), medium (default standard parameters), high (maximum update frequency/unthrottled accuracy)"
    )
    
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Specify the SQLite database path (optional)"
    )
    
    parser.add_argument(
        "--limit",
        action="store_true",
        help="Limit execution to 50 frames for testing purposes."
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        default="general",
        choices=["general", "detailed"],
        help="Display log detail levels: general (displays INFO) or detailed (displays DEBUG)."
    )
    
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Disable JSON database saving to reduce disk I/O bottleneck."
    )

    args = parser.parse_args()

    # FIX: args.mode is already evaluated to a DisplayMode Enum by argparse type validator
    selected_mode = args.mode

    config_kwargs = {}
    if args.db:
        config_kwargs["data_store"] = args.db
        
    config = LepauteConfig(**config_kwargs)
    
    # FIX: args.perf is already evaluated to a PerformanceMode Enum by argparse type validator
    config.performance_mode = args.perf

    selected_log_level = logging.DEBUG if args.log_level == "detailed" else logging.INFO

    logging.basicConfig(level=selected_log_level, format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
    logger.info(f"[System] System booting in mode: {selected_mode.name} with performance profile: {config.performance_mode.name}")
    logger.info(f"[System] Log level set to: {args.log_level.upper()}")
    run_unlimited = not args.limit
    
    run_pipeline(
        config, 
        display_mode=selected_mode, 
        unlimited=run_unlimited, 
        save_json=not args.no_save,
        mock=False
    )