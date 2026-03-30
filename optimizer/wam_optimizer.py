"""
WAM Codec Self-Improving Optimizer
==================================
Evaluates candidate WAM configurations against a benchmark attack suite.
Keeps configurations that improve median z-score; discards failures.

Run:
    python -m optimizer.wam_optimizer
"""

import sys
import os
import json
import math
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image

from attacks import crop, resize, rotate, flip, jpeg, noise, blur, brightness, contrast, saturation
from core import WatermarkManager
from roco_core import decode_from_bits
from roco_ecc import decode_with_ecc
from watermark_anything.data.metrics import msg_predict_inference


# Constants
TOTAL_BITS = 32
DEFAULT_NUM_IMAGES = 20
DEFAULT_LOG_FILE = 'optimizer/improvement_log.json'


def compute_z_score(correct_bits: int, total_bits: int = TOTAL_BITS) -> float:
    """Compute z-score for bit accuracy."""
    expected = total_bits / 2
    std = math.sqrt(total_bits / 4)
    if std < 1e-9:
        return 0.0
    return (correct_bits - expected) / std


# Attack suite: (name, function)
ATTACKS = [
    ('clean', lambda a: a),
    ('jpeg_75', lambda a: jpeg(a, 75)),
    ('jpeg_50', lambda a: jpeg(a, 50)),
    ('noise_20', lambda a: noise(a, 20)),
    ('crop_75', lambda a: crop(a, 0.75)),
    ('crop_50', lambda a: crop(a, 0.50)),
    ('resize_05', lambda a: resize(a, 0.5)),
    ('rotate_15', lambda a: rotate(a, 15)),
    ('blur_5', lambda a: blur(a, 5)),
]


class WAMVariant:
    """Self-contained WAM configuration with tunable parameters."""
    
    def __init__(
        self,
        scaling_w: float = 2.0,
        scaling_i: float = 1.0,
        label: str = "variant",
    ):
        self.scaling_w = scaling_w
        self.scaling_i = scaling_i
        self.label = label
        self._original_sw: Optional[float] = None
        self._original_si: Optional[float] = None
    
    def apply(self, manager: WatermarkManager) -> None:
        """Apply this variant's parameters to the WatermarkManager."""
        self._original_sw = manager.wam.scaling_w
        self._original_si = manager.wam.scaling_i
        manager.wam.scaling_w = self.scaling_w
        manager.wam.scaling_i = self.scaling_i
    
    def restore(self, manager: WatermarkManager) -> None:
        """Restore original parameters."""
        if self._original_sw is not None:
            manager.wam.scaling_w = self._original_sw
        if self._original_si is not None:
            manager.wam.scaling_i = self._original_si
    
    def to_dict(self) -> dict:
        return {
            'label': self.label,
            'scaling_w': self.scaling_w,
            'scaling_i': self.scaling_i,
        }


class WAMOptimizer:
    """Self-improving optimizer for WAM watermarking."""
    
    def __init__(
        self,
        num_test_images: int = DEFAULT_NUM_IMAGES,
        log_file: str = DEFAULT_LOG_FILE,
    ):
        self.manager = WatermarkManager()
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.test_image_files = self._generate_test_images(num_test_images)
        self.test_messages = ['ABC', 'XYZ', 'QWE', 'RST', 'UVW']
        self.results: List[Dict[str, Any]] = []
        
        # Accept thresholds
        self.accept_delta = 0.10
        self.accept_wins = 2
        
        # Best configuration
        self.best_variant = WAMVariant(label="baseline")
        self.best_z = 0.0
        self.best_median_z: Dict[str, float] = {}
    
    def _generate_test_images(self, num_images: int) -> List[str]:
        """Generate random test images and save to temp files."""
        temp_dir = tempfile.mkdtemp(prefix='wam_opt_')
        files = []
        
        for i in range(num_images):
            np.random.seed(42 + i)
            arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            path = os.path.join(temp_dir, f'test_{i}.png')
            img.save(path)
            files.append(path)
        
        return files
    
    def _benchmark_single_image(
        self, 
        img_file: str, 
        message: str
    ) -> Dict[str, List[float]]:
        """Benchmark a single image against all attacks."""
        attack_z_scores: Dict[str, List[float]] = {name: [] for name, _ in ATTACKS}
        
        img_tensor, _, coords = self.manager.embed(img_file, message, 'corners')
        
        for attack_name, attack_fn in ATTACKS:
            attacked = attack_fn(img_tensor)
            result = self.manager.verify_tensor(attacked, message)
            z = compute_z_score(result['correct_bits'], TOTAL_BITS)
            attack_z_scores[attack_name].append(z)
        
        return attack_z_scores
    
    def benchmark_variant(self, variant: WAMVariant) -> dict:
        """Benchmark a variant against all attacks."""
        start_time = time.time()
        attack_z_scores: Dict[str, List[float]] = {name: [] for name, _ in ATTACKS}
        
        variant.apply(self.manager)
        
        try:
            for img_idx, img_file in enumerate(self.test_image_files):
                message = self.test_messages[img_idx % len(self.test_messages)]
                img_z_scores = self._benchmark_single_image(img_file, message)
                
                for attack_name, z_list in img_z_scores.items():
                    attack_z_scores[attack_name].extend(z_list)
        finally:
            variant.restore(self.manager)
        
        # Compute median z-scores per attack
        median_z = {name: float(np.median(z_scores)) for name, z_scores in attack_z_scores.items()}
        overall_median = float(np.median(list(median_z.values())))
        
        elapsed = time.time() - start_time
        
        return {
            'variant': variant.to_dict(),
            'median_z': median_z,
            'overall_median_z': overall_median,
            'elapsed_seconds': elapsed,
        }
    
    def _initial_candidates(self) -> List[WAMVariant]:
        """Generate initial candidate variants."""
        return [
            WAMVariant(label="baseline", scaling_w=2.0, scaling_i=1.0),
            WAMVariant(label="sw_30", scaling_w=3.0, scaling_i=1.0),
            WAMVariant(label="sw_15", scaling_w=1.5, scaling_i=1.0),
            WAMVariant(label="sw_25", scaling_w=2.5, scaling_i=1.0),
            WAMVariant(label="sw_25_si09", scaling_w=2.5, scaling_i=0.9),
            WAMVariant(label="sw_20_si08", scaling_w=2.0, scaling_i=0.8),
            WAMVariant(label="sw_35", scaling_w=3.5, scaling_i=1.0),
            WAMVariant(label="sw_40", scaling_w=4.0, scaling_i=1.0),
            WAMVariant(label="sw_10", scaling_w=1.0, scaling_i=1.0),
            WAMVariant(label="sw_05", scaling_w=0.5, scaling_i=1.0),
        ]
    
    def _continuation_candidates(self, winner: WAMVariant) -> List[WAMVariant]:
        """Generate continuation candidates from a winner."""
        candidates = []
        
        # Fine sweeps on scaling_w
        for delta in [0.25, 0.5, -0.25, -0.5]:
            new_sw = winner.scaling_w + delta
            if 0.5 <= new_sw <= 5.0:
                candidates.append(WAMVariant(
                    label=f"sw_{new_sw:.2f}",
                    scaling_w=new_sw,
                    scaling_i=winner.scaling_i,
                ))
        
        # Fine sweeps on scaling_i
        for delta in [0.1, -0.1]:
            new_si = winner.scaling_i + delta
            if 0.5 <= new_si <= 1.5:
                candidates.append(WAMVariant(
                    label=f"si_{new_si:.2f}",
                    scaling_w=winner.scaling_w,
                    scaling_i=new_si,
                ))
        
        return candidates
    
    def _evaluate_variant(self, variant: WAMVariant, round_num: int, total: int) -> bool:
        """Evaluate a variant and return True if accepted."""
        print(f"\n  Round {round_num}/{total}: {variant.label}")
        result = self.benchmark_variant(variant)
        
        delta = result['overall_median_z'] - self.best_z
        wins = sum(1 for k, v in result['median_z'].items() 
                  if v > self.best_median_z.get(k, 0))
        
        # Accept if improves by delta threshold AND wins on enough attacks
        # Or accept first variant if no baseline yet
        accept = (delta >= self.accept_delta and wins >= self.accept_wins) or self.best_z == 0
        
        if accept:
            self.best_z = result['overall_median_z']
            self.best_variant = variant
            self.best_median_z = result['median_z']
            print(f"    ✓ ACCEPTED: z={result['overall_median_z']:.2f} (delta={delta:+.2f}, wins={wins})")
        else:
            print(f"    ✗ REJECTED: z={result['overall_median_z']:.2f} (delta={delta:+.2f}, wins={wins})")
        
        self.results.append(result)
        return accept
    
    def run(self, rounds: int = 10, generations: int = 5) -> WAMVariant:
        """Run the optimization loop."""
        print(f"WAM Optimizer - {len(self.test_image_files)} test images")
        print(f"Testing {len(ATTACKS)} attack types")
        print("=" * 60)
        
        # Initial rounds
        print(f"\n[Initial Rounds: {rounds}]")
        candidates = self._initial_candidates()[:rounds]
        
        for i, variant in enumerate(candidates):
            self._evaluate_variant(variant, i + 1, rounds)
        
        # Continuation generations
        for gen in range(generations):
            print(f"\n[Generation {gen+1}/{generations}]")
            candidates = self._continuation_candidates(self.best_variant)
            
            for i, variant in enumerate(candidates):
                self._evaluate_variant(variant, i + 1, len(candidates))
        
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print(f"Best z-score: {self.best_z:.2f}")
        print(f"Best variant: {self.best_variant.label}")
        print(f"Params: {self.best_variant.to_dict()}")
        
        self._save_log()
        
        return self.best_variant
    
    def _save_log(self) -> None:
        """Save results to log file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {self.log_file}")


if __name__ == '__main__':
    optimizer = WAMOptimizer(
        num_test_images=10,
        log_file='optimizer/improvement_log.json',
    )
    
    best = optimizer.run(rounds=5, generations=2)
