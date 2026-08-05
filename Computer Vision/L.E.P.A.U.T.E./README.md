# Lie Equivariant Perception Algebraic Unified Transform Embedding Framework (L.E.P.A.U.T.E. Framework)

## Abstract

The L.E.P.A.U.T.E. Framework is a geometrically unified computer vision and robotics environment engineered for real-time spatial telemetry perception and multi-task optimization. By mapping sequential visual transitions into explicit Lie group representations, specifically the Special Euclidean Group SE(3), the architecture enforces strict structural equivariance across frame-to-frame transformations. Operating on continuous monocular video feeds, the core execution pipeline combines direct image alignment tracking, transformer-based structural embeddings, and zero-shot open-vocabulary vision classifiers to resolve high-fidelity camera ego-motion and relative object poses within a decoupled, asynchronous, multi-threaded framework.

## System Overview

The framework decouples data ingestion, asynchronous neural network inference execution, and state-based sequence persistence to maximize operational throughput and prevent main-thread latency blockages.

```mermaid
graph TD
    %% Data Ingestion and Loop Setup
    A[CameraIOStream / Parametric Failover] -->|Frame RGB & Metadata| B[Main Pipeline Loop]
    B -->|Worker Health Verification| C{worker.is_healthy?}
    C -->|No: Timeout/Crash| D[SYSTEM HALT: Prevent Zombie Lock]
    
    %% Async Polling & Tracker Execution
    C -->|Yes| E[Poll worker.get_latest_resolved_state]
    E -->|Check Buffer State| F{Async State Resolved?}
    F -->|Yes: pop State| G["Flag: has_async = True<br/>Extract Category & Confidence"]
    F -->|No: Cache Miss| H[Flag: has_async = False]
    
    G --> I[Execute MonocularDirectTracker.track]
    H --> I
    
    %% Direct Tracker Internal Path
    I -->|PyTorch Photometric Alignment| J{Direct Alignment Score >= 0.15?}
    J -->|No| K["Execute _execute_orb_pnp_fallback<br/>SolvePnPRansac Local Opt"]
    
    %% Job Dispatched to Worker Queue
    I -->|If Track Score > 0.1| L[worker.enqueue_job]
    K --> L
    
    %% Mode Arbitration Matrix
    G --> M{Arbitration Matrix}
    J --> M
    K --> M
    
    M -->|has_async == True| N["Mode: Refined<br/>best_rel_xi = refined_xi_rel"]
    M -->|has_async == False && score >= 0.1| O["Mode: Tracker<br/>best_rel_xi = tracker_xi_rel"]
    M -->|Tracking Alignment Fails| P["Mode: Recovery<br/>Query ManifoldKinematicForecaster"]
    
    %% Async Worker Thread Loop
    subgraph Async Background Execution [InferenceWorker Thread Context]
        L --> Q[("job_queue: Max 5")]
        Q -->|De-queue Task| R[YOLOClassifier Zero-Shot Head]
        R -->|Predict Label Space| S[Dynamic Scale Prior Mapping]
        S --> T[SE3ResidualRefiner Forward Pass]
        
        %% Refiner Internal Graph
        subgraph SE3ResidualRefiner Pipeline
            T --> U[ResNet18 Base Features]
            U --> V[Lightweight depth_head Map]
            U --> W[Inject Spatial pos_embed & pose_emb]
            W --> X[SE3CrossAttentionBlock Space]
            X --> Y[Joint Residual & Uncertainty Regression]
        end
        
        Y -->|"delta_xi, delta_scale, unc_pose, unc_scale"| Z["Staging Stage: history_buffer"]
    end
    
    Z -->|Atomic .pop eviction| E
    
    %% State Fusion & Propagation
    N --> AA["SE(3) Exponential Mapping & Poses Composition"]
    O --> AA
    P --> AA
    
    AA --> BB["Update ManifoldKinematicForecaster State<br/>Log-Space Scale Velocity Estimation"]
    BB --> CC[Update Global 2D Trajectory Vector]
    CC --> DD[SequenceDataCollector / SQLite WAL Persistence]
    CC --> EE[OpenCV HUD Context Render]
    
    %% Feedback Loop
    BB -.->|Feedback Scale Prior| I
```

## Features and Capabilities

* **Lie Algebra Structural Equivariance**: Direct integration of Special Euclidean Group SE(3) tracking components utilizing customized geometric differentiable layers (`se3_exp_map`, `se3_log_map`, `compose_poses`) to warp raw visual feature maps relative to computed coordinate transitions.
* **Asynchronous Multi-Process Inference Ring**: High-frequency data stream frames are entirely decoupled from deep learning execution blocks using worker queues, atomic state-locks, and a dynamic Time-To-Live (TTL) history buffer to completely prevent tracking thread interface latency. The `InferenceWorker` runs in an isolated OS process to bypass the Python GIL.
* **Hybrid Multi-Mode Pose Fusion**: A robust tracking arbitration layer that dynamically transitions between refined deep learning outputs (`SE3ResidualRefiner`), direct Gauss-Newton intensity-alignment tracking (`MonocularDirectTracker`), and a manifold-based kinematic forecasting recovery loop (`ManifoldKinematicForecaster`) during periods of severe visual degradation or occlusion.
* **Open-Vocabulary Target Classification**: Embedded YOLO-based detection powered by Ultralytics, enabling runtime target tracking definitions and automatic physical dimension scale prior injection without requiring visual categorization head retraining.
* **Robust Hardware Failover Ingestion**: The data acquisition interface (`CameraIOStream`) implements an automated fallback sequence, instantly spawning a synthetic parametric image sequence generator if the targeted physical capture hardware encounters latency anomalies or connection drops.
* **Deterministic Training Grounding**: Complete optimization infrastructure embedding geometric residual regression models with seed replication and validation partitioning for reproducible offline training passes of the `SE3ResidualRefiner`.
* **Apple Silicon MPS Safety**: Thread-safe context managers (`mps_safe`) and explicit synchronization locks ensure stable Metal Performance Shaders execution under concurrent inference workloads.
* **Monocular Scale Resolution**: Dynamic scale prior injection combined with residual scale correction and log-space velocity estimation resolves metric ambiguity inherent to monocular vision.

## License

MIT License.

## Contributors

### PyCon HK 2025

* Primary contributor: [shz2](https://twitter.com/shivvor2)
* Special thanks to: [BenBenCHAK](https://github.com/BenBenCHAK), [usertam](https://github.com/usertam)

## References

This project is a implementation of [Wu, C. (2025). Lie Equivariant Perception Algebraic Unified Transform Embedding Framework (L.E.P.A.U.T.E. Framework): Achieving Precise Modeling of Geometric Transformations. Document Identification Code: 20250501_01.](https://github.com/dev1virtuoso/Documentation/blob/main/dev1virtuoso/Research/2025/05_2025/20250501/20250501_01.md)

[Dataset](https://huggingface.co/datasets/dev1virtuoso/lepaute-dataset)