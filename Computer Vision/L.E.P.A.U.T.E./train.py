import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import random_split, Dataset

try:
    from pipeline_and_config import LepauteConfig, EquivariantDataset, train_sequence_loop
    from models import SE3ResidualRefiner
except ImportError as e:
    print(f"CRITICAL: Failed to import LEPAUTE infrastructure components: {e}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] LEPAUTE_TRAIN: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LEPAUTE.OfflineTraining")


def set_reproducibility_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_data_splits(dataset: Dataset, split_ratio: float = 0.2) -> Tuple[Dataset, Optional[Dataset]]:
    dataset_size = len(dataset)
    if dataset_size < 8:
        logger.warning(
            f"Dataset size ({dataset_size}) is insufficient for validation partitioning. "
            f"Deploying entire allocation to the training track."
        )
        return dataset, None
    
    val_len = int(dataset_size * split_ratio)
    train_len = dataset_size - val_len
    
    generator = torch.Generator().manual_seed(42)
    return random_split(dataset, [train_len, val_len], generator=generator)


def load_manifests_from_dir(json_dir: Path) -> List[Dict]:
    records = []
    if not json_dir.is_dir():
        logger.error(f"Directory Error: Expected JSON directory not found at {json_dir}")
        sys.exit(1)
        
    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        logger.error(f"Dataset manifest loading failed: No JSON files found in {json_dir}")
        sys.exit(1)
        
    logger.info(f"Discovered {len(json_files)} individual JSON metadata files in {json_dir}. Compiling...")
    
    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                record = json.load(f)
                records.append(record)
        except json.JSONDecodeError as jde:
            logger.warning(f"Corruption Detected: Failed parsing structured json sequence file {jf}: {jde}")
        except Exception as e:
            logger.warning(f"File Access Error: Encountered failure attempting to open target data payload at {jf}: {e}")
            
    if not records:
        logger.error(f"Execution Aborted: Could not load any valid records from {json_dir}")
        sys.exit(1)
        
    return records


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Production Orchestration for training LEPAUTE SE(3) Subsystems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset_dir", type=str, required=True, help="Root path containing the train/ and test/ directories")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum number of processing epochs.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Target path for state saving.")
    parser.add_argument("--device", type=str, default=None, help="Force override configuration compute target (e.g., 'cuda', 'mps', 'cpu').")
    parser.add_argument("--no_compile", action="store_true", help="Disable PyTorch 2.x torch.compile() optimization step.")
    parser.add_argument("--seed", type=int, default=42, help="Stochastic initialization seed value.")
    
    parser.add_argument(
        "--resume_mode", 
        type=str, 
        choices=["resume", "scratch", "ask"], 
        default="ask", 
        help="Training continuation strategy: 'resume' automatically loads the latest progress, 'scratch' forces retraining from scratch, 'ask' prompts interactively in the terminal at runtime."
    )
    
    args = parser.parse_args()

    set_reproducibility_seeds(args.seed)

    data_root = Path(args.dataset_dir)
    ckpt_root = Path(args.checkpoint_dir)

    try:
        ckpt_root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"System IO Error: Unable to create checkpoint destination directory {ckpt_root}: {e}")
        sys.exit(1)
    
    latest_ckpt_path = ckpt_root / "latest_checkpoint.pth"
    resume_flag = False
    
    if latest_ckpt_path.exists():
        if args.resume_mode == "ask":
            if sys.stdin.isatty():
                try:
                    choice = input("\n[Progress Prompt] Detected previous training progress snapshot (latest_checkpoint.pth).\nDo you want to resume training from the last checkpoint? [Y/n]: ").strip().lower()
                    if choice in ["", "y", "yes"]:
                        resume_flag = True
                        logger.info("User decision: Resuming training flow from the previous checkpoint.")
                    else:
                        resume_flag = False
                        logger.info("User decision: Starting a completely new training session (existing history files will be overwritten subsequently).")
                except (KeyboardInterrupt, EOFError):
                    logger.warning("\nInteractive input abnormally interrupted. Failsafe mechanism activated: Defaulting to resuming from previous progress.")
                    resume_flag = True
            else:
                logger.info("Non-interactive terminal execution detected (Headless environment). Bypassing standard prompt and automatically resuming.")
                resume_flag = True
        elif args.resume_mode == "resume":
            resume_flag = True
            logger.info("Command line argument configuration: Forced resumption of training from the previous checkpoint.")
        else:
            resume_flag = False
            logger.info("Command line argument configuration: Forced history wipe, starting a completely new training session from scratch.")
    else:
        logger.info("No previous training checkpoints detected. Automatically initiating a completely new optimization pipeline.")
        resume_flag = False

    config = LepauteConfig()
    
    if config.device == "mps":
        logger.warning("Apple Silicon MPS detected. Disabling torch.compile due to MSL generator instability.")
        config.use_compiler = False
    
    if args.device is not None:
        config.device = args.device
    if args.no_compile:
        config.use_compiler = False

    logger.info(f"LEPAUTE Engine Configuration Initialized. Targets: Device={config.device} | Compiler={config.use_compiler}")

    train_dir = data_root / "train"
    test_dir = data_root / "test"
    
    train_set, val_set = None, None

    if train_dir.is_dir() and test_dir.is_dir():
        logger.info("Detected explicitly separated 'train' and 'test' dataset partitions.")
        
        train_records = load_manifests_from_dir(train_dir)
        test_records = load_manifests_from_dir(test_dir)
        
        logger.info(f"Compiled Training Split: {len(train_records)} instances. Compiled Validation Split: {len(test_records)} instances.")
        
        try:
            train_set = EquivariantDataset(data_list=train_records, config=config, data_dir=str(train_dir))
            val_set = EquivariantDataset(data_list=test_records, config=config, data_dir=str(test_dir))
        except Exception as e:
            logger.error(f"Initialization Exception: Data structure initialization failure within EquivariantDataset: {e}")
            sys.exit(1)
            
    else:
        logger.info("Explicit splits not found. Defaulting to single-volume random partitioning strategy.")
        
        if not list(data_root.glob("*.json")) and not list(data_root.glob("*.jpg")):
            logger.error(f"Execution Aborted: Expected valid files not found in dataset root directory: {data_root}")
            sys.exit(1)
            
        stored_records = load_manifests_from_dir(data_root)
        logger.info(f"Single-volume dataset successfully compiled. Found {len(stored_records)} active monocular sequence blocks.")
        
        try:
            full_dataset = EquivariantDataset(data_list=stored_records, config=config, data_dir=str(data_root))
        except Exception as e:
            logger.error(f"Initialization Exception: Data structure initialization failure within EquivariantDataset: {e}")
            sys.exit(1)
            
        train_set, val_set = get_data_splits(full_dataset)

    logger.info("Instantiating Deep SE(3) Residual Refiner Subsystem architecture.")
    try:
        model = SE3ResidualRefiner(config=config)
    except Exception as e:
        logger.error(f"Architecture Generation Error: Failed to construct neural network graph layout: {e}")
        sys.exit(1)
    
    logger.info(f"Optimization track initialized. Syncing checkpoints output path to: {ckpt_root.resolve()}")
    
    try:
        train_loss, val_loss = train_sequence_loop(
            model=model,
            train_dataset=train_set,
            val_dataset=val_set,
            config=config,
            epochs=args.epochs,
            checkpoint_dir=str(ckpt_root),
            resume=resume_flag
        )
        logger.info(f"Optimization Sequence Concluded Successfully. Training Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}")
        
    except KeyboardInterrupt:
        logger.warning("Optimization sequence execution forcibly interrupted by operational operator signal (SIGINT). Clean exit enforced.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Runtime Operational Critical Failure: Execution context interrupted during deep tracking loop: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()