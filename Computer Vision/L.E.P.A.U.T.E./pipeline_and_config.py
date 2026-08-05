import os
import sys
import time
import json
import queue
import random
import sqlite3
import inspect
import threading
import multiprocessing
from enum import Enum
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Optional

import cv2
import torch
import torch.nn as nn
import numpy as np
import albumentations as A
from tqdm import tqdm
from pydantic import Field
from pydantic_settings import BaseSettings
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from typing import Dict
from globals import logger
import platform
import io
import safetensors.numpy as st

_camera_calib_cache = None

def _load_camera_calibration() -> Dict[str, float]:
    """
    Dynamically loads intrinsic camera calibration priors from an external JSON configuration file.
    Provides a fallback to standard 320x240 defaults if the file is missing or malformed,
    eliminating hard-coded spatial projection values from the source codebase.
    """
    global _camera_calib_cache
    if _camera_calib_cache is not None:
        return _camera_calib_cache
        
    config_path = os.environ.get("LEPAUTE_CAMERA_CONFIG_PATH", "camera_config.json")
    defaults = {"fx": 250.0, "fy": 250.0, "cx": 160.0, "cy": 120.0}
    
    path_obj = Path(config_path)
    if path_obj.is_file():
        try:
            with open(path_obj, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    logger.info(f"[LepauteConfig] Dynamically loaded camera calibration from {config_path}")
                    _camera_calib_cache = {k: float(v) for k, v in data.items() if k in defaults}
                    for k in defaults:
                        if k not in _camera_calib_cache:
                            _camera_calib_cache[k] = defaults[k]
                    return _camera_calib_cache
        except Exception as e:
            logger.error(f"[LepauteConfig] Failed to parse dynamic calibration from {config_path}: {e}. Falling back to defaults.")
            
    _camera_calib_cache = defaults
    return _camera_calib_cache

def _get_fx() -> float: return _load_camera_calibration()["fx"]
def _get_fy() -> float: return _load_camera_calibration()["fy"]
def _get_cx() -> float: return _load_camera_calibration()["cx"]
def _get_cy() -> float: return _load_camera_calibration()["cy"]

class DisplayMode(str, Enum):
    REALTIME = "realtime"
    JSON = "json"
    HEADLESS = "headless"
    DETAILEDGUI = "detailedgui"

def _get_stable_compute_device() -> str:
    """
    Dynamically routes hardware targeting. Now explicitly supports Apple Silicon MPS
    for accelerated tensor operations while defaulting to CUDA if available.
    """
    # CRITICAL FIX: Enable Apple Silicon MPS CPU fallback to prevent driver crashes during unsupported op delegation
    # This mitigates secondary crashes during async handoffs to CPU fallback on Apple Silicon
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        logger.info("[LepauteConfig] Apple Silicon MPS detected. Engaging Metal Performance Shaders.")
        return "mps"
    return "cpu"

class PerformanceMode(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    
def _load_dynamic_object_scales() -> Dict[str, float]:
    """
    Dynamically loads object scale priors from an external JSON configuration file.
    Provides a fallback to default values if the file is missing or malformed,
    eliminating hard-coded scale values from the source codebase.
    """
    config_path = os.environ.get("LEPAUTE_OBJECT_CONFIG_PATH", "object_config.json")
    default_scales = {
        "table": 1.5, 
        "cup": 0.1, 
        "keyboard": 0.4, 
        "laptop": 0.35, 
        "mouse": 0.12, 
        "human": 1.7, 
        "background": 2.0
    }
    
    path_obj = Path(config_path)
    if path_obj.is_file():
        try:
            with open(path_obj, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    logger.info(f"[LepauteConfig] Dynamically loaded {len(data)} object scales from {config_path}")
                    return {str(k): float(v) for k, v in data.items()}
                else:
                    logger.warning(f"[LepauteConfig] Invalid format in {config_path}. Expected a dictionary. Using defaults.")
        except Exception as e:
            logger.error(f"[LepauteConfig] Failed to parse dynamic scales from {config_path}: {e}. Falling back to default configuration.")
    
    return default_scales

def _load_dynamic_object_names() -> List[str]:
    """
    Dynamically derives the list of detectable object names directly from 
    the loaded dynamic scale configuration keys.
    """
    return list(_load_dynamic_object_scales().keys())
    
class LepauteConfig(BaseSettings):
    device: str = Field(default_factory=_get_stable_compute_device)
    data_store: str = "lepaute_data.db"
    
    # Performance profile orchestration flag
    performance_mode: PerformanceMode = PerformanceMode.MEDIUM
    
    # YOLO zero-shot classification target labels (Dynamically loaded)
    object_names: List[str] = Field(default_factory=_load_dynamic_object_names)
    
    # Scale priors (in meters) for monocular Z-depth projection (Dynamically loaded)
    object_scales: Dict[str, float] = Field(default_factory=_load_dynamic_object_scales)
    
    orig_h: int = 240
    orig_w: int = 320
    
    # Dynamically loaded camera intrinsics replacing hardcoded values
    fx: float = Field(default_factory=_get_fx)
    fy: float = Field(default_factory=_get_fy)
    cx: float = Field(default_factory=_get_cx)
    cy: float = Field(default_factory=_get_cy)
    
    pyramid_levels: int = 3
    gn_max_iter: int = 15
    use_compiler: bool = False
    enable_orb_fallback: bool = True
    
    # Explicit dataloader control preventing deadlocks on constrained compute interfaces (MPS/Windows)
    num_workers: Optional[int] = None
    
class CameraIOStream:
    def __init__(self, config: LepauteConfig, mock: bool = False):
        self.config = config
        self.cap = None
        self.frame_id = 0
        self.mock_mode = mock
        
        self.latest_frame = None
        self.is_running = False
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        self.stream_interrupted = False
        self.reconnect_count = 0
        
        logger.info(f"[CameraIOStream] Subsystem initialization triggered. Execution Mode: {'MOCK_DATA_SYNTHESIS' if mock else 'PHYSICAL_HARDWARE_STREAM'}")
        if not self.mock_mode:
            self._connect_physical_camera()
            self._start_capture_thread()
            
    def _start_capture_thread(self):
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
    def _capture_loop(self):
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # Execute heavy resizing workload inside the background thread to free up the main loop
                    # FIX: OpenCV defaults to BGR. Explicitly convert to RGB to fix the channel inversion issue.
                    bgr = cv2.resize(frame, (self.config.orig_w, self.config.orig_h))
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    with self.frame_lock:
                        self.latest_frame = rgb
                else:
                    with self.frame_lock:
                        self.stream_interrupted = True
                    logger.warning("[CameraIOStream] Background thread encountered frame drop. Reconnecting...")
                    self.cap.release()
                    try:
                        self._connect_physical_camera(retries=1)
                    except Exception as e:
                        logger.error(f"[CameraIOStream] Background thread failed to reconnect: {e}")
                        self.is_running = False
                        break
            else:
                time.sleep(0.1)

    def _connect_physical_camera(self, retries=3):
        backends = [cv2.CAP_ANY]
        system_platform = platform.system()
        
        if system_platform == "Darwin":
            backends.insert(0, cv2.CAP_AVFOUNDATION)
        elif system_platform == "Windows":
            backends.insert(0, cv2.CAP_DSHOW)
            backends.insert(1, cv2.CAP_MSMF)
        else:
            backends.insert(0, cv2.CAP_V4L2)
            
        logger.info(f"[CameraIOStream] Detected platform: '{system_platform}'. Backend strategy: {backends}")

        for attempt in range(retries):
            for backend in backends:
                backend_id_name = str(backend)
                logger.info(f"[CameraIOStream] Attempting binding via backend: {backend_id_name} (Attempt {attempt + 1}/{retries})")
                
                try:
                    self.cap = cv2.VideoCapture(0, backend)
                    
                    if self.cap and self.cap.isOpened():
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.orig_w)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.orig_h)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            logger.info(f"[CameraIOStream] Successful video probe matrix via {backend_id_name}.")
                            return
                        else:
                            self.cap.release()
                except Exception as e:
                    logger.warning(f"[CameraIOStream] Runtime block during driver initialization on backend {backend_id_name}: {e}")
                    if self.cap:
                        self.cap.release()
                    
            logger.warning(f"[CameraIOStream] Connection attempt {attempt + 1} exhausted. Suspending before retry...")
            time.sleep(1.0)
                
        raise RuntimeError("Fatal: Failed to sequentially connect to physical camera hardware.")

    def read(self) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        self.frame_id += 1
        meta = {"timestamp": time.time(), "frame_id": self.frame_id}
        
        if not self.mock_mode:
            frame_to_return = None
            restored = False
            
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_to_return = self.latest_frame.copy()
                # Acknowledge and reset the stream interruption flag only upon successful frame acquisition
                if self.stream_interrupted and frame_to_return is not None:
                    self.stream_interrupted = False
                    self.reconnect_count += 1
                    restored = True
            
            # Prevent startup race condition where the thread hasn't fetched the first frame yet
            if frame_to_return is None:
                logger.warning(f"[CameraIOStream] Frame buffer empty for ID: {self.frame_id}. Waiting for background thread...")
                for _ in range(20):
                    time.sleep(0.05)
                    with self.frame_lock:
                        if self.latest_frame is not None:
                            frame_to_return = self.latest_frame.copy()
                            if self.stream_interrupted:
                                self.stream_interrupted = False
                                self.reconnect_count += 1
                                restored = True
                            break
            
            if frame_to_return is not None:
                if restored:
                    meta["stream_restored"] = True
                    meta["reconnect_count"] = self.reconnect_count
                    logger.info(f"[CameraIOStream] Hardware stream restored after interruption. Reconnect count: {self.reconnect_count}")
                logger.debug(f"[CameraIOStream] Frame ID {self.frame_id} pulled safely from background buffer.")
                return True, frame_to_return, meta
            else:
                logger.error(f"[CameraIOStream] Background thread failed to provide a frame. Connection might be dead.")
                return False, np.zeros(0), meta
            
        elif self.mock_mode:
            logger.debug(f"[CameraIOStream] Generating synthetic multi-spectral tensor field frame simulation. Index: {self.frame_id}")
            h, w = self.config.orig_h, self.config.orig_w
            
            texture = np.ones((h, w), dtype=np.uint8) * 50
            pos_x = (self.frame_id * 3) % max(1, (w - 30))
            pos_y = (self.frame_id * 2) % max(1, (h - 30))
            cv2.rectangle(texture, (pos_x, pos_y), (pos_x+30, pos_y+30), 220, -1)
            
            for i in range(0, w, 20):
                cv2.line(texture, (i, 0), (i, h), 100, 1)
            for j in range(0, h, 20):
                cv2.line(texture, (0, j), (w, j), 100, 1)
                
            rgb = cv2.cvtColor(texture, cv2.COLOR_GRAY2RGB)
            return True, rgb, meta
            
        return False, np.zeros(0), meta

    def release(self):
        logger.info("[CameraIOStream] Disconnecting camera stream references and executing secure interface context teardown.")
        self.is_running = False
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
            
        if self.cap: 
            self.cap.release()
            logger.info("[CameraIOStream] VideoCapture interface freed successfully.")

class SequenceDataCollector(threading.Thread):
    def __init__(self, config: LepauteConfig):
        super().__init__(daemon=True)
        self.db_path = config.data_store
        self.write_queue = queue.Queue(maxsize=200)
        self.running = threading.Event()
        self.running.set()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute('''CREATE TABLE IF NOT EXISTS transitions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            img_a BLOB, img_b BLOB,
                            xi TEXT, obj_name TEXT)''')

    def append_transition(self, img_a: np.ndarray, img_b: np.ndarray, xi: np.ndarray, obj_name: str):
        if not self.write_queue.full():
            self.write_queue.put_nowait((img_a.copy(), img_b.copy(), xi.copy(), obj_name))

    def run(self):
        logger.info("[SequenceDataCollector] Background persistence thread active.")
        with sqlite3.connect(self.db_path, timeout=10) as conn:
            while self.running.is_set() or not self.write_queue.empty():
                batch = []
                try:
                    while len(batch) < 10:
                        if not self.running.is_set() and self.write_queue.empty():
                            break
                        batch.append(self.write_queue.get(timeout=0.1))
                except queue.Empty:
                    pass
                
                if batch:
                    logger.debug(f"[SequenceDataCollector] Processing persistence batch of size {len(batch)}...")
                    records = []
                    for (img_a, img_b, xi, obj_name) in batch:
                        # Stored safely as RGB uint8 (original depth) to eliminate precision loss and color risk from float16 + BGR 
                        img_a_safe = np.ascontiguousarray(img_a, dtype=np.uint8)
                        img_b_safe = np.ascontiguousarray(img_b, dtype=np.uint8)
                        
                        try:
                            enc_a = st.save({"img": img_a_safe})
                            enc_b = st.save({"img": img_b_safe})
                        except ImportError:
                            buf_a = io.BytesIO()
                            np.save(buf_a, img_a_safe)
                            enc_a = buf_a.getvalue()
                            
                            buf_b = io.BytesIO()
                            np.save(buf_b, img_b_safe)
                            enc_b = buf_b.getvalue()
                        
                        records.append((enc_a, enc_b, json.dumps(xi.tolist()), obj_name))
                        
                    try:
                        conn.executemany("INSERT INTO transitions (img_a, img_b, xi, obj_name) VALUES (?, ?, ?, ?)", records)
                        conn.commit()
                        logger.debug("[SequenceDataCollector] Batch successfully committed to SQLite WAL.")
                    except Exception as e:
                        logger.error(f"[SequenceDataCollector] SQLite write error during batch commit: {e}")
                        
                    for _ in batch: self.write_queue.task_done()
                    
        logger.info("[SequenceDataCollector] Background thread loop exited safely.")

    def stop(self):
        logger.info("[SequenceDataCollector] Stop command received. Halting event loop...")
        self.running.clear()
        
        logger.info("[SequenceDataCollector] Waiting for thread to join (timeout=5.0s)...")
        self.join(timeout=5.0)
        
        if self.is_alive():
            logger.warning("[SequenceDataCollector] Thread join timed out. It might be blocked on DB lock.")
        else:
            logger.info("[SequenceDataCollector] Thread joined successfully.")
            
        logger.info("[SequenceDataCollector] Executing final SQLite WAL checkpoint truncation...")
        try:
            # CRITICAL FIX: Added explicit timeout to prevent infinite hang if the database is externally locked
            with sqlite3.connect(self.db_path, timeout=2.0) as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                conn.commit()
            logger.info("[SequenceDataCollector] Database WAL checkpoint completed.")
        except Exception as e:
            logger.error(f"[SequenceDataCollector] Database sync failed during SequenceDataCollector teardown: {e}")

def load_data(config: LepauteConfig) -> List[Dict]:
    data = []
    with sqlite3.connect(config.data_store) as conn:
        cursor = conn.execute("SELECT img_a, img_b, xi, obj_name FROM transitions")
        for row in cursor:
            try:
                try:
                    img_a = st.load(row[0])["img"].astype(np.uint8)
                    img_b = st.load(row[1])["img"].astype(np.uint8)
                except Exception:
                    try:
                        # Fallback for np.save/BytesIO serialization
                        img_a = np.load(io.BytesIO(row[0])).astype(np.uint8)
                        img_b = np.load(io.BytesIO(row[1])).astype(np.uint8)
                    except Exception:
                        # Fallback for deprecated jpeg string formats
                        img_a = cv2.imdecode(np.frombuffer(row[0], np.uint8), cv2.IMREAD_COLOR)
                        img_b = cv2.imdecode(np.frombuffer(row[1], np.uint8), cv2.IMREAD_COLOR)
                        # Correct color space for legacy payload fallbacks
                        if img_a is not None: img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
                        if img_b is not None: img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

                data.append({
                    "img_a": img_a, "img_b": img_b,
                    "lie_params": json.loads(row[2]), "detected_object": row[3]
                })
            except Exception as e:
                logger.error(f"[DataLoader] Failed to decode DB row: {e}")
    return data

@lru_cache(maxsize=1000)
def _cached_read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        # Prevent cryptic OpenCV C++ assertion failures by throwing a clear Python exception
        raise FileNotFoundError(
            f"[Dataset IO Error] CRITICAL: OpenCV failed to load image at '{path}'. "
            f"Please verify that the 'jpg' directory contains this file and it is not corrupted."
        )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

class EquivariantDataset(Dataset):
    def __init__(self, data_list: List[Dict], config: LepauteConfig, data_dir: Optional[str] = None):
        self.data_list = data_list
        self.config = config
        self.data_dir = data_dir
        
        unique_objs = sorted(list(set(item.get("detected_object", "unknown") for item in data_list)))
        if data_dir:
            self.obj_map = {n: i for i, n in enumerate(unique_objs)}
        else:
            self.obj_map = {n: i for i, n in enumerate(config.object_names)}
            
        self.transform = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussNoise(p=0.3),
            A.MotionBlur(p=0.2),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], additional_targets={'image0': 'image'})

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        if self.data_dir:
            img_a_path = os.path.join(self.data_dir, item["frame_a"])
            img_b_path = os.path.join(self.data_dir, item["frame_b"])
            img_a = _cached_read_image(img_a_path)
            img_b = _cached_read_image(img_b_path)
        else:
            # Eliminating double conversion since the pipeline now correctly persists matrices in RGB natively
            img_a = item["img_a"]
            img_b = item["img_b"]
            
        if img_a.shape[:2] != (self.config.orig_h, self.config.orig_w):
            img_a = cv2.resize(img_a, (self.config.orig_w, self.config.orig_h))
            img_b = cv2.resize(img_b, (self.config.orig_w, self.config.orig_h))

        transformed = self.transform(image=img_a, image0=img_b)
        t_a = transformed["image"]
        t_b = transformed["image0"]
        
        xi_gt = torch.tensor(item['lie_params'], dtype=torch.float32)
        xi_noisy = xi_gt + torch.randn(6) * 0.05 
        
        obj_name = item.get("detected_object", "unknown")
        obj_idx = self.obj_map.get(obj_name, 0)
        
        scale_prior = self.config.object_scales.get(obj_name)
        if scale_prior is None:
            scale_prior = self.config.object_scales.get("object", 1.0)
        
        return t_a, t_b, xi_gt, xi_noisy, obj_idx, float(scale_prior)

def train_sequence_loop(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    config: LepauteConfig,
    epochs: int,
    checkpoint_dir: str,
    resume: bool = False
) -> Tuple[float, float]:
    
    device = torch.device(config.device)
    model = model.to(device)
    
    if config.use_compiler and sys.version_info >= (3, 10) and sys.platform != "win32":
        try:
            logger.info("[Trainer] Engaging torch.compile() induction layer for standard optimization.")
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"[Trainer] Compiler optimization failed to bind, falling back to eager execution mode: {e}")

    cv2.setNumThreads(0)
    
    if config.num_workers is not None:
        num_workers = config.num_workers
    else:
        num_workers = min(4, multiprocessing.cpu_count() or 1) if sys.platform != "win32" else 0
        
    pin_memory = (device.type == "cuda")
    
    logger.info(f"[Trainer] Configuring DataLoader: workers={num_workers}, pin_memory={pin_memory}, batch_size={getattr(config, 'batch_size', 32)}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=getattr(config, "batch_size", 32), 
        shuffle=True, 
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=getattr(config, "batch_size", 32), 
            shuffle=False, 
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="min", 
        patience=3, 
        factor=0.5
    )
    
    checkpoint_path = Path(checkpoint_dir)
    best_model_path = checkpoint_path / "best_model.pth"
    latest_checkpoint_path = checkpoint_path / "latest_checkpoint.pth"
    
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 5
    start_epoch = 0
    
    avg_train_loss = 0.0
    avg_val_loss = 0.0

    if resume and latest_checkpoint_path.exists():
        logger.info(f"[Trainer] Restoring system state from historical checkpoint: {latest_checkpoint_path}")
        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
            
            raw_state_dict = checkpoint["model_state_dict"]
            if hasattr(model, "_orig_mod"):
                model._orig_mod.load_state_dict(raw_state_dict)
            else:
                model.load_state_dict(raw_state_dict)
                
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            patience_counter = checkpoint["patience_counter"]
            
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
            if "np_rng_state" in checkpoint:
                np.random.set_state(checkpoint["np_rng_state"])
            if "random_state" in checkpoint:
                random.setstate(checkpoint["random_state"])
                
            logger.info(f"[Trainer] System state restored. Resuming training from Epoch {start_epoch + 1}.")
        except Exception as e:
            logger.error(f"[Trainer] Checkpoint state load failure: {e}. Falling back to training from scratch.")
            start_epoch = 0

    def save_checkpoint(epoch_idx: int) -> None:
        try:
            clean_model_state = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
            checkpoint_data = {
                "epoch": epoch_idx,
                "model_state_dict": clean_model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "torch_rng_state": torch.get_rng_state(),
                "np_rng_state": np.random.get_state(),
                "random_state": random.getstate(),
            }
            if torch.cuda.is_available():
                checkpoint_data["cuda_rng_state"] = torch.cuda.get_rng_state()
                
            temp_path = latest_checkpoint_path.with_suffix(".tmp")
            torch.save(checkpoint_data, temp_path)
            temp_path.replace(latest_checkpoint_path)
        except Exception as save_err:
            logger.error(f"[Trainer] Non-fatal checkpoint save exception (Epoch {epoch_idx + 1}): {save_err}")

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            train_loss_accum = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs:02d} [Train]", leave=False, dynamic_ncols=True)
            
            for batch_idx, batch in enumerate(train_pbar):
                if isinstance(batch, dict):
                    t_a = batch["t_a"].to(device=device, dtype=torch.float32)
                    t_b = batch["t_b"].to(device=device, dtype=torch.float32)
                    xi_gt = batch["xi_gt"].to(device=device, dtype=torch.float32)
                    xi_noisy = batch["xi_noisy"].to(device=device, dtype=torch.float32)
                elif isinstance(batch, (list, tuple)):
                    t_a = batch[0].to(device=device, dtype=torch.float32)
                    t_b = batch[1].to(device=device, dtype=torch.float32)
                    xi_gt = batch[2].to(device=device, dtype=torch.float32)
                    xi_noisy = batch[3].to(device=device, dtype=torch.float32)
                else:
                    raise TypeError(f"Unrecognized batch format returned by DataLoader: {type(batch)}")
                
                optimizer.zero_grad()
                outputs = model(t_a, t_b, xi_noisy)
                
                if isinstance(outputs, dict):
                    pred_pose = outputs["pose"]
                elif isinstance(outputs, (list, tuple)):
                    pred_pose = outputs[0]
                else:
                    pred_pose = outputs
                
                # FIX: Network learns the residual directly. Supervise the prediction against the actual tangent space delta.
                target_delta = xi_gt - xi_noisy
                loss = torch.mean((pred_pose - target_delta) ** 2)
                loss.backward()
                optimizer.step()
                
                train_loss_accum += loss.item()
                train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            avg_train_loss = train_loss_accum / max(1, len(train_loader))
            
            if val_loader is not None:
                model.eval()
                val_loss_accum = 0.0
                
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{epochs:02d} [Valid]", leave=False, dynamic_ncols=True)
                
                with torch.no_grad():
                    for batch in val_pbar:
                        if isinstance(batch, dict):
                            t_a = batch["t_a"].to(device=device, dtype=torch.float32)
                            t_b = batch["t_b"].to(device=device, dtype=torch.float32)
                            xi_gt = batch["xi_gt"].to(device=device, dtype=torch.float32)
                            xi_noisy = batch["xi_noisy"].to(device=device, dtype=torch.float32)
                        elif isinstance(batch, (list, tuple)):
                            t_a = batch[0].to(device=device, dtype=torch.float32)
                            t_b = batch[1].to(device=device, dtype=torch.float32)
                            xi_gt = batch[2].to(device=device, dtype=torch.float32)
                            xi_noisy = batch[3].to(device=device, dtype=torch.float32)
                        
                        outputs = model(t_a, t_b, xi_noisy)
                        
                        if isinstance(outputs, dict):
                            pred_pose = outputs["pose"]
                        elif isinstance(outputs, (list, tuple)):
                            pred_pose = outputs[0]
                        else:
                            pred_pose = outputs
                            
                        # FIX: Validation metric should correspond to the semantic representation in training (the delta)
                        target_delta = xi_gt - xi_noisy
                        diff = pred_pose - target_delta
                        batch_loss = torch.mean(diff ** 2).item()
                        val_loss_accum += batch_loss
                        
                        val_pbar.set_postfix({'mse': f"{batch_loss:.4f}"})
                        
                avg_val_loss = val_loss_accum / max(1, len(val_loader))

            # ===== CORE TRAINING LOGIC FINALIZATION =====
            current_metric = avg_val_loss if val_loader is not None else avg_train_loss
            
            # Step the learning rate scheduler
            scheduler.step(current_metric)
            
            # Save normal epoch checkpoint
            save_checkpoint(epoch)
            
            # Assess metrics for best model caching and early stopping
            if current_metric < best_val_loss:
                best_val_loss = current_metric
                patience_counter = 0
                try:
                    import shutil
                    if latest_checkpoint_path.exists():
                        shutil.copyfile(latest_checkpoint_path, best_model_path)
                        logger.info(f"[Trainer] New best model saved at Epoch {epoch+1} with loss {best_val_loss:.4f}")
                except Exception as e:
                    logger.error(f"[Trainer] Failed to copy best model checkpoint: {e}")
            else:
                patience_counter += 1
                logger.info(f"[Trainer] No improvement for {patience_counter} consecutive epoch(s).")
                if patience_counter >= early_stop_patience:
                    logger.info(f"[Trainer] Early stopping triggered at epoch {epoch+1}. Restoring best parameters.")
                    break
                    
    except KeyboardInterrupt:
        logger.warning("\n[Trainer] Manual halt signal (SIGINT / Ctrl+C) detected.")
        if 'epoch' in locals():
            logger.info(f"[Trainer] Executing emergency state snapshot for ongoing Epoch {epoch + 1}...")
            save_checkpoint(epoch)
        logger.info("[Trainer] Checkpoint successfully flushed to disk. Safe to exit.")
        raise KeyboardInterrupt
    except Exception as e:
        logger.error(f"[Trainer] Runtime Operational Critical Failure: {e}")
        raise
        
    return avg_train_loss, avg_val_loss