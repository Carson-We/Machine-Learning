import os
import time
import json
import sqlite3
import tempfile
import unittest
from collections import deque
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from geometry import (
    skew_symmetric,
    se3_exp_map,
    se3_log_map,
    compose_poses,
)
from vision_tracking import (
    MonocularDirectTracker,
    YOLOClassifier,
    ManifoldKinematicForecaster,
)
from models import (
    SE3ResidualRefiner,
    MonocularSE3Warping,
    SE3CrossAttentionBlock,
)
from globals import logger, _mps_lock
from pipeline_and_config import (
    LepauteConfig,
    DisplayMode,
    PerformanceMode,
    EquivariantDataset,
    SequenceDataCollector,
    train_sequence_loop,
    load_data,
    CameraIOStream,
)
from main import run_pipeline


class TestLepauteCoreArchitecture(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.config = LepauteConfig(
            device="cpu",
            data_store=os.path.join(self.test_dir.name, "test_store.db"),
            orig_h=64, 
            orig_w=64,
            fx=50.0, 
            fy=50.0, 
            cx=32.0, 
            cy=32.0,
            pyramid_levels=2,
            enable_orb_fallback=False,
            use_compiler=False
        )

    def tearDown(self):
        self.test_dir.cleanup()

    def test_skew_symmetric_properties(self):
        v = torch.tensor([[1.5, -2.3, 4.1]], dtype=torch.float32)
        K = skew_symmetric(v)
        
        self.assertEqual(K.shape, (1, 3, 3))
        self.assertTrue(torch.allclose(K, -K.transpose(1, 2), atol=1e-6))

    def test_se3_manifold_invariants(self):
        zero_xi = torch.zeros(1, 6, dtype=torch.float32)
        T_identity = se3_exp_map(zero_xi)
        self.assertTrue(torch.allclose(T_identity[:, :3, :3], torch.eye(3).unsqueeze(0)))
        self.assertTrue(torch.allclose(T_identity[:, :3, 3], torch.zeros(1, 3)))
        
        small_xi = torch.tensor([[1e-5, -2e-5, 1e-5, 3e-5, -1e-5, 2e-5]], dtype=torch.float32)
        T_small = se3_exp_map(small_xi)
        recovered_small_xi = se3_log_map(T_small)
        self.assertTrue(torch.allclose(small_xi, recovered_small_xi, atol=1e-5))
        
        large_xi = torch.tensor([[0.2, -0.1, 0.5, 0.1, -0.2, 0.3]], dtype=torch.float32)
        T_large = se3_exp_map(large_xi)
        recovered_large_xi = se3_log_map(T_large)
        self.assertTrue(torch.allclose(large_xi, recovered_large_xi, atol=1e-4))
        
        tiny_xi = torch.tensor([[5e-6, -2e-5, 8e-6, 5e-4, -7e-4, 1e-4]], dtype=torch.float32)
        T_tiny = se3_exp_map(tiny_xi)
        recovered_tiny_xi = se3_log_map(T_tiny)
        self.assertTrue(torch.allclose(tiny_xi, recovered_tiny_xi, atol=1e-6))

        pi_xi = torch.tensor([[0.1, -0.2, 0.3, 3.14159, 0.0, 0.0]], dtype=torch.float32)
        T_pi = se3_exp_map(pi_xi)
        recovered_pi_xi = se3_log_map(T_pi)
        self.assertFalse(torch.isnan(recovered_pi_xi).any())
        self.assertTrue(torch.allclose(pi_xi[:3], recovered_pi_xi[:3], atol=1e-3))

    def test_compose_poses(self):
        T1 = se3_exp_map(torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.1, 0.0]], dtype=torch.float32))
        T2 = se3_exp_map(torch.tensor([[0.0, 0.2, 0.0, 0.0, 0.0, 0.2]], dtype=torch.float32))
        
        T_composed = compose_poses(T1, T2)
        self.assertEqual(T_composed.shape, (1, 4, 4))

    def test_gauss_newton_pyramid_stability_with_scale(self):
        tracker = MonocularDirectTracker(self.config)
        img1 = np.ones((64, 64, 3), dtype=np.uint8) * 128
        cv2.circle(img1, (32, 32), 16, (64, 64, 64), -1)
        
        scale_prior = self.config.object_scales.get("laptop", 0.35)
        xi, score = tracker.track(img1, img1, scale_prior=scale_prior)
        
        self.assertEqual(xi.shape, (6, ))
        self.assertTrue(np.allclose(xi, 0.0, atol=1e-2))
        self.assertTrue(0.0 <= score <= 1.0)

    def test_hybrid_orb_fallback_execution(self):
        self.config.enable_orb_fallback = True
        tracker = MonocularDirectTracker(self.config)
        
        img_a = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.rectangle(img_a, (10, 10), (25, 25), (255, 255, 255), -1)
        cv2.circle(img_a, (45, 45), 10, (255, 255, 255), -1)
        cv2.line(img_a, (5, 50), (25, 55), (255, 255, 255), 2)
        
        img_b = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.rectangle(img_b, (12, 10), (27, 25), (255, 255, 255), -1)
        cv2.circle(img_b, (47, 45), 10, (255, 255, 255), -1)
        cv2.line(img_b, (7, 50), (27, 55), (255, 255, 255), 2)
        
        xi, score = tracker.track(img_a, img_b, scale_prior=1.0)
        self.assertEqual(xi.shape, (6, ))
        self.assertTrue(0.0 <= score <= 1.0)

    @patch('vision_tracking.YOLO')
    def test_yolo_classifier_mocked_inference(self, mock_yolo_init):
        mock_model = MagicMock()
        mock_model.names = {0: 'table', 1: 'cup'}
        mock_yolo_init.return_value = mock_model

        mock_result = MagicMock()
        mock_result.names = {0: 'table', 1: 'cup'}
        
        mock_result.probs = None
        
        mock_boxes = MagicMock()
        mock_boxes.conf = torch.tensor([0.85, 0.95])
        mock_boxes.cls = torch.tensor([1.0, 0.0]) 
        mock_boxes.__len__.return_value = 2
        
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        
        classifier = YOLOClassifier(self.config)
        dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
        
        label, score = classifier.predict(dummy_img)
        
        self.assertEqual(label, "table")
        self.assertAlmostEqual(score, 0.95, places=5)

    def test_monocular_se3_warping(self):
        warper = MonocularSE3Warping(self.config)
        img_tensor = torch.rand(2, 3, 64, 64, dtype=torch.float32)
        xi_tensor = torch.zeros(2, 6, dtype=torch.float32)
        scale_tensor = torch.ones(2, dtype=torch.float32)
        
        warped_img, valid_mask = warper(img_tensor, xi_tensor, scale_tensor)
        self.assertEqual(warped_img.shape, (2, 3, 64, 64))
        self.assertEqual(valid_mask.shape, (2, 1, 64, 64))

    def test_se3_cross_attention_block(self):
        block = SE3CrossAttentionBlock(dim=32, num_heads=2)
        visual_feat_flat = torch.rand(2, 256, 32, dtype=torch.float32)
        
        output = block(query=visual_feat_flat, key=visual_feat_flat, value=visual_feat_flat)
        self.assertEqual(output.shape, (2, 256, 32))

    def test_se3_residual_refiner_and_compilation_loading(self):
        refiner = SE3ResidualRefiner(config=self.config, feature_dim=256, max_resolution=64)
        img_a = torch.rand(2, 3, 64, 64, dtype=torch.float32)
        img_b = torch.rand(2, 3, 64, 64, dtype=torch.float32)
        xi_noisy = torch.rand(2, 6, dtype=torch.float32)
        
        delta_xi, delta_scale, unc_pose, unc_scale = refiner(img_a, img_b, xi_noisy)
        
        self.assertEqual(delta_xi.shape, (2, 6))
        self.assertEqual(delta_scale.shape, (2, 1))
        self.assertEqual(unc_pose.shape, (2, 6))
        self.assertEqual(unc_scale.shape, (2, 1))
        
        state_dict = refiner.state_dict()
        compiled_state_dict = {f"_orig_mod.{k}": v for k, v in state_dict.items()}
        load_status = refiner.load_compiled_state_dict(compiled_state_dict)
        self.assertIsNotNone(load_status)

    def test_camera_io_stream_mock(self):
        stream = CameraIOStream(self.config, mock=True)
        ret, frame, meta = stream.read()
        
        self.assertTrue(ret)
        self.assertEqual(frame.shape, (64, 64, 3))
        self.assertIn("timestamp", meta)
        self.assertEqual(meta["frame_id"], 1)
        stream.release()

    def test_concurrent_collector_schema(self):
        collector = SequenceDataCollector(config=self.config)
        collector.start()
        
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        xi = np.array([0.1, -0.2, 0.3, 0.0, 0.1, -0.5], dtype=np.float32)
        
        collector.append_transition(img, img, xi, "keyboard")
        collector.stop()
        
        loaded = load_data(self.config)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["detected_object"], "keyboard")
        self.assertEqual(len(loaded[0]["lie_params"]), 6)

    def test_equivariant_dataset_initialization(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_data = [{
            "img_a": img, 
            "img_b": img,
            "lie_params": [0.1] * 6, 
            "detected_object": "cup"
        }]
        
        ds = EquivariantDataset(mock_data, self.config, data_dir=None)
        t_a, t_b, xi_gt, xi_noisy, obj_idx, scale_prior = ds[0]
        
        self.assertEqual(t_a.shape, (3, 64, 64))
        self.assertEqual(t_b.shape, (3, 64, 64))
        self.assertEqual(xi_gt.shape, (6, ))
        self.assertEqual(xi_noisy.shape, (6, ))
        self.assertIsInstance(scale_prior, float)

    def test_manifold_kinematic_forecaster(self):
        forecaster = ManifoldKinematicForecaster()
        measured_pose = np.eye(4)
        delta_xi = np.zeros(6, dtype=np.float32)
        delta_scale = 1.05
        
        forecaster.update_state(measured_pose, delta_xi, delta_scale, timestamp=1.0)
        predicted_pose, predicted_scale = forecaster.predict(timestamp=2.0)
        
        self.assertEqual(predicted_pose.shape, (4, 4))
        self.assertIsInstance(predicted_scale, float)
        self.assertGreater(predicted_scale, 1.0)

    def test_train_sequence_loop_execution(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_data = [
            {"img_a": img, "img_b": img, "lie_params": [0.0]*6, "detected_object": "mouse"},
            {"img_a": img, "img_b": img, "lie_params": [0.0]*6, "detected_object": "mouse"}
        ]
        
        train_ds = EquivariantDataset(mock_data, self.config)
        val_ds = EquivariantDataset(mock_data, self.config)
        refiner = SE3ResidualRefiner(config=self.config)
        
        orig_dataloader_init = DataLoader.__init__
        def safe_dataloader_init(self, dataset, batch_size=1, shuffle=False, *args, **kwargs):
            kwargs['num_workers'] = 0
            kwargs['persistent_workers'] = False
            orig_dataloader_init(self, dataset, batch_size=batch_size, shuffle=shuffle, *args, **kwargs)
            
        with patch.object(DataLoader, '__init__', safe_dataloader_init), \
             tempfile.TemporaryDirectory() as ckpt_dir:
             
            train_loss, val_loss = train_sequence_loop(
                model=refiner,
                train_dataset=train_ds,
                val_dataset=val_ds,
                config=self.config,
                epochs=1,
                checkpoint_dir=ckpt_dir
            )
            
            self.assertIsInstance(train_loss, float)
            self.assertIsInstance(val_loss, float)

    def test_integration_moving_sequence(self):
        tracker = MonocularDirectTracker(self.config)
        
        base_img = np.zeros((64, 64, 3), dtype=np.uint8)
        frames = []
        for i in range(5):
            img = base_img.copy()
            cv2.rectangle(img, (20 + i*2, 20), (40 + i*2, 40), (255, 255, 255), -1)
            frames.append(img)
            
        cumulative_xi = np.zeros(6)
        for i in range(1, len(frames)):
            xi, score = tracker.track(frames[i-1], frames[i], scale_prior=1.0)
            cumulative_xi += xi
            
        self.assertNotEqual(cumulative_xi[0], 0.0)

    @patch('main.cv2.imshow')
    @patch('main.cv2.waitKey', return_value=-1)
    def test_benchmark_performance_modes(self, mock_wait, mock_imshow):
        
        self.config.performance_mode = PerformanceMode.LOW
        start_low = time.time()
        res_low = run_pipeline(self.config, display_mode=DisplayMode.HEADLESS, unlimited=False, save_json=False, mock=True)
        time_low = time.time() - start_low
        
        self.config.performance_mode = PerformanceMode.HIGH
        start_high = time.time()
        res_high = run_pipeline(self.config, display_mode=DisplayMode.HEADLESS, unlimited=False, save_json=False, mock=True)
        time_high = time.time() - start_high
        
        self.assertTrue(len(res_low) > 0)
        self.assertTrue(len(res_high) > 0)
        
        self.assertGreater(time_low, time_high)

    def test_long_duration_drift(self):
        forecaster = ManifoldKinematicForecaster()
        
        current_pose = np.eye(4)
        for i in range(100):
            noisy_delta_xi = np.random.normal(0, 1e-4, 6).astype(np.float32)
            noisy_delta_scale = max(0.99, min(1.01, 1.0 + np.random.normal(0, 1e-4)))
            
            forecaster.update_state(
                measured_pose=current_pose, 
                delta_xi=noisy_delta_xi, 
                delta_scale=noisy_delta_scale, 
                timestamp=float(i) * 0.033, 
                weight=0.5
            )
            
            predicted_pose, predicted_scale = forecaster.predict(float(i+1) * 0.033)
            current_pose = predicted_pose
            
        final_xi = se3_log_map(torch.from_numpy(current_pose).unsqueeze(0).float())
        
        self.assertTrue(torch.all(torch.abs(final_xi) < 0.1).item())
        self.assertAlmostEqual(forecaster.get_scale(), 1.0, places=1)


if __name__ == "__main__":
    unittest.main()