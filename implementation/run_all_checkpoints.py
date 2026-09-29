"""
run_all_checkpoints.py — Master Execution & Verification Runner with Persistent Detailed Time Logs
Paper: Quantum Information Entanglement Index for Facial Expression Analysis (ECML PKDD 2026)

Executes all 7 implementation stages end-to-end with high-precision microsecond timing:
- Stage 1: Linear Algebra & Quantum Mathematical Engine Core
- Stage 2: 68-Point Facial Landmark Detection (MediaPipe / OpenCV / Anatomical fallback)
- Stage 3: 7-Region Anatomical Slicing & 4-Extreme-Point Normalization
- Stage 4: Quantum Information Entanglement Engine (Intra 3-qubit, Inter 5-qubit)
- Stage 5: 24D QIEI Descriptor Builder & FACS Action Unit Mapping
- Stage 6: Real CK+ Dataset Loader, Subject-Disjoint 80:10:10 Split & Batch Feature Extraction
- Stage 7: Benchmark Evaluator (SVM, RF, MLP), Paired t-test & Figure 5 Radar Plot Generator

All execution logs and timing metrics are automatically persisted into:
  - implementation/logs/run_YYYYMMDD_HHMMSS.log
  - implementation/logs/run_YYYYMMDD_HHMMSS_timings.json
  - implementation/logs/latest_run.log
  - implementation/logs/latest_timings.json
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from logger import ExecutionLogger
from stage1_linalg_core import run_stage1_checkpoint
from stage2_landmark_detector import run_stage2_checkpoint
from stage3_region_partitioner import run_stage3_checkpoint
from stage4_quantum_engine import run_stage4_checkpoint
from stage5_descriptor_builder import run_stage5_checkpoint
from stage6_dataset_manager import run_stage6_checkpoint
from stage7_benchmark_evaluator import run_stage7_checkpoint


def run_master_verification(num_dataset_samples: int = 50):
    logger = ExecutionLogger("qiei_end_to_end_verification")
    
    stages = [
        ("STAGE-1", "Linear Algebra & Quantum Gate Invariants", run_stage1_checkpoint),
        ("STAGE-2", "68-Point Facial Landmark Detection", run_stage2_checkpoint),
        ("STAGE-3", "7-Region Anatomical Slicing & Diamonds", run_stage3_checkpoint),
        ("STAGE-4", "Quantum Entanglement Engine (Intra/Inter)", run_stage4_checkpoint),
        ("STAGE-5", "24D Descriptor Assembly & FACS Mapping", run_stage5_checkpoint),
        ("STAGE-6", "Real CK+ Dataset & Subject Split", lambda: run_stage6_checkpoint(max_samples_for_test=num_dataset_samples)),
        ("STAGE-7", "Benchmark Evaluator & Figure 5 Radar", run_stage7_checkpoint)
    ]
    
    all_passed = True
    for stage_id, title, func in stages:
        logger.start_stage(stage_id, title)
        t0 = time.perf_counter()
        try:
            func()
            duration = time.perf_counter() - t0
            logger.end_stage(status="PASSED", metadata={"duration_sec": duration})
        except Exception as e:
            duration = time.perf_counter() - t0
            all_passed = False
            logger.error(f"Exception during {stage_id}: {e}")
            logger.end_stage(status=f"FAILED", metadata={"error": str(e), "duration_sec": duration})
            
    summary = logger.finalize()
    return all_passed, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QIEI reproduction suite with detailed timing logs")
    parser.add_argument("--samples", type=int, default=50, help="Number of CK+ samples for batch feature extraction stage")
    args = parser.parse_args()
    
    success, _ = run_master_verification(num_dataset_samples=args.samples)
    sys.exit(0 if success else 1)
