"""
Lambda³ GPU性能ベンチマークスイート
GPU実装の性能評価とボトルネック解析
"""

import numpy as np
import cupy as cp
import time
import psutil
import GPUtil
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
from contextlib import contextmanager
import json
import os

from ..analysis.md_lambda3_detector_gpu import MDLambda3DetectorGPU, MDConfig
from ..analysis.two_stage_analyzer_gpu import TwoStageAnalyzerGPU, ResidueAnalysisConfig


@dataclass
class BenchmarkResult:
    """ベンチマーク結果"""
    test_name: str
    n_frames: int
    n_atoms: int
    cpu_time: float
    gpu_time: float
    speedup: float
    gpu_memory_used: float
    gpu_memory_peak: float
    cpu_memory_used: float
    gpu_utilization: float
    throughput_fps: float
    error: Optional[str] = None


class Lambda3BenchmarkSuite:
    """Lambda³ GPU ベンチマークスイート"""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.gpu_available = self._check_gpu()
        self.results = []
    
    def _check_gpu(self) -> bool:
        """GPU利用可能性チェック"""
        try:
            cp.cuda.Device()
            print("✓ GPU available:", cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
            return True
        except:
            print("✗ GPU not available, running CPU-only tests")
            return False
    
    @contextmanager
    def _gpu_memory_monitor(self):
        """GPUメモリ使用量モニタリング"""
        if self.gpu_available:
            cp.get_default_memory_pool().free_all_blocks()
            mempool = cp.get_default_memory_pool()
            
            start_used = mempool.used_bytes()
            peak_used = start_used
            
            yield lambda: mempool.used_bytes()
            
            end_used = mempool.used_bytes()
            peak_used = mempool.total_bytes()
            
            self.gpu_memory_stats = {
                'used': end_used - start_used,
                'peak': peak_used
            }
        else:
            yield lambda: 0
            self.gpu_memory_stats = {'used': 0, 'peak': 0}
    
    def run_all_benchmarks(self):
        """全ベンチマーク実行"""
        print("\n🏁 Starting Lambda³ GPU Benchmark Suite")
        print("="*60)
        
        # 1. 基本性能テスト
        self.benchmark_basic_performance()
        
        # 2. スケーラビリティテスト
        self.benchmark_scalability()
        
        # 3. メモリ効率テスト
        self.benchmark_memory_efficiency()
        
        # 4. コンポーネント別性能
        self.benchmark_components()
        
        # 5. 2段階解析性能
        self.benchmark_two_stage_analysis()
        
        # 6. 精度検証
        self.benchmark_accuracy()
        
        # レポート生成
        self.generate_report()
    
    def benchmark_basic_performance(self):
        """基本性能ベンチマーク"""
        print("\n📊 Basic Performance Benchmark")
        print("-"*40)
        
        test_sizes = [
            (1000, 100, "Small"),
            (5000, 500, "Medium"),
            (10000, 1000, "Large"),
            (50000, 2000, "XLarge")
        ]
        
        config = MDConfig()
        config.use_extended_detection = True
        config.use_phase_space = False  # 速度重視
        
        for n_frames, n_atoms, size_name in test_sizes:
            print(f"\n  Testing {size_name}: {n_frames} frames, {n_atoms} atoms")
            
            # ダミートラジェクトリ生成
            trajectory = self._generate_test_trajectory(n_frames, n_atoms)
            
            # GPU版
            if self.gpu_available:
                gpu_time = self._benchmark_gpu(trajectory, config)
            else:
                gpu_time = None
            
            # CPU版
            cpu_time = self._benchmark_cpu(trajectory, config)
            
            # 結果記録
            if gpu_time:
                speedup = cpu_time / gpu_time if cpu_time else 0
                throughput = n_frames / gpu_time
            else:
                speedup = 1.0
                throughput = n_frames / cpu_time if cpu_time else 0
            
            result = BenchmarkResult(
                test_name=f"basic_{size_name}",
                n_frames=n_frames,
                n_atoms=n_atoms,
                cpu_time=cpu_time,
                gpu_time=gpu_time or 0,
                speedup=speedup,
                gpu_memory_used=self.gpu_memory_stats.get('used', 0) / 1e9,
                gpu_memory_peak=self.gpu_memory_stats.get('peak', 0) / 1e9,
                cpu_memory_used=psutil.Process().memory_info().rss / 1e9,
                gpu_utilization=self._get_gpu_utilization(),
                throughput_fps=throughput
            )
            
            self.results.append(result)
            self._print_result(result)
    
    def benchmark_scalability(self):
        """スケーラビリティテスト"""
        print("\n\n📈 Scalability Benchmark")
        print("-"*40)
        
        # 固定原子数、フレーム数を変化
        n_atoms = 1000
        frame_counts = [100, 500, 1000, 5000, 10000, 20000]
        
        config = MDConfig()
        config.gpu_batch_size = 5000
        
        scalability_results = []
        
        for n_frames in frame_counts:
            print(f"\n  Testing {n_frames} frames...")
            
            trajectory = self._generate_test_trajectory(n_frames, n_atoms)
            
            if self.gpu_available:
                start = time.time()
                with self._gpu_memory_monitor():
                    detector = MDLambda3DetectorGPU(config, device='gpu')
                    _ = detector.analyze(trajectory)
                gpu_time = time.time() - start
                
                throughput = n_frames / gpu_time
                scalability_results.append({
                    'n_frames': n_frames,
                    'time': gpu_time,
                    'throughput': throughput,
                    'memory': self.gpu_memory_stats['used'] / 1e9
                })
                
                print(f"    Time: {gpu_time:.2f}s, Throughput: {throughput:.1f} fps")
        
        # スケーラビリティプロット
        if scalability_results:
            self._plot_scalability(scalability_results)
    
    def benchmark_memory_efficiency(self):
        """メモリ効率ベンチマーク"""
        print("\n\n💾 Memory Efficiency Benchmark")
        print("-"*40)
        
        if not self.gpu_available:
            print("  GPU not available, skipping memory tests")
            return
        
        # メモリ制限テスト
        n_frames = 10000
        atom_counts = [500, 1000, 2000, 4000, 8000]
        
        memory_results = []
        
        for n_atoms in atom_counts:
            print(f"\n  Testing {n_atoms} atoms...")
            
            trajectory = self._generate_test_trajectory(n_frames, n_atoms)
            data_size_gb = trajectory.nbytes / 1e9
            
            try:
                with self._gpu_memory_monitor() as get_memory:
                    detector = MDLambda3DetectorGPU()
                    
                    # メモリ使用量を定期的に記録
                    memory_timeline = []
                    
                    start = time.time()
                    result = detector.analyze(trajectory)
                    end = time.time()
                    
                    memory_results.append({
                        'n_atoms': n_atoms,
                        'data_size_gb': data_size_gb,
                        'gpu_memory_gb': self.gpu_memory_stats['used'] / 1e9,
                        'memory_ratio': self.gpu_memory_stats['used'] / trajectory.nbytes,
                        'time': end - start,
                        'success': True
                    })
                    
                    print(f"    Data size: {data_size_gb:.2f} GB")
                    print(f"    GPU memory: {self.gpu_memory_stats['used']/1e9:.2f} GB")
                    print(f"    Memory ratio: {self.gpu_memory_stats['used']/trajectory.nbytes:.2f}")
                    
            except Exception as e:
                memory_results.append({
                    'n_atoms': n_atoms,
                    'data_size_gb': data_size_gb,
                    'error': str(e),
                    'success': False
                })
                print(f"    ❌ Failed: {str(e)}")
        
        self._plot_memory_efficiency(memory_results)
    
    def benchmark_components(self):
        """コンポーネント別性能テスト"""
        print("\n\n🔧 Component Performance Benchmark")
        print("-"*40)
        
        # テストデータ
        n_frames = 5000
        n_atoms = 1000
        trajectory = self._generate_test_trajectory(n_frames, n_atoms)
        
        components = [
            ('MD Features', self._benchmark_md_features),
            ('Lambda Structures', self._benchmark_lambda_structures),
            ('Anomaly Detection', self._benchmark_anomaly_detection),
            ('Boundary Detection', self._benchmark_boundary_detection),
            ('Extended Detection', self._benchmark_extended_detection)
        ]
        
        component_times = {}
        
        for name, benchmark_func in components:
            print(f"\n  {name}...")
            
            if self.gpu_available:
                gpu_time = benchmark_func(trajectory, use_gpu=True)
                component_times[f"{name}_GPU"] = gpu_time
                print(f"    GPU: {gpu_time:.3f}s")
            
            cpu_time = benchmark_func(trajectory, use_gpu=False)
            component_times[f"{name}_CPU"] = cpu_time
            print(f"    CPU: {cpu_time:.3f}s")
            
            if self.gpu_available:
                speedup = cpu_time / gpu_time
                print(f"    Speedup: {speedup:.1f}x")
        
        self._plot_component_performance(component_times)
    
    def benchmark_two_stage_analysis(self):
        """2段階解析の性能テスト"""
        print("\n\n🔬 Two-Stage Analysis Benchmark")
        print("-"*40)
        
        # テストケース
        n_frames = 10000
        n_atoms = 2000
        n_residues = 100
        trajectory = self._generate_test_trajectory(n_frames, n_atoms)
        
        # マクロ解析
        print("\n  Stage 1: Macro analysis...")
        config = MDConfig()
        detector = MDLambda3DetectorGPU(config)
        
        start = time.time()
        macro_result = detector.analyze(trajectory)
        macro_time = time.time() - start
        
        print(f"    Macro analysis time: {macro_time:.2f}s")
        
        # 残基解析
        print("\n  Stage 2: Residue analysis...")
        events = [
            (1000, 2000, 'event1'),
            (3000, 4000, 'event2'),
            (6000, 7000, 'event3')
        ]
        
        residue_config = ResidueAnalysisConfig()
        analyzer = TwoStageAnalyzerGPU(residue_config)
        
        start = time.time()
        residue_result = analyzer.analyze_trajectory(
            trajectory, macro_result, events, n_residues
        )
        residue_time = time.time() - start
        
        print(f"    Residue analysis time: {residue_time:.2f}s")
        print(f"    Total time: {macro_time + residue_time:.2f}s")
        
        # 並列性のテスト
        print("\n  Testing parallel event processing...")
        residue_config.parallel_events = True
        
        start = time.time()
        parallel_result = analyzer.analyze_trajectory(
            trajectory, macro_result, events, n_residues
        )
        parallel_time = time.time() - start
        
        print(f"    Parallel time: {parallel_time:.2f}s")
        print(f"    Parallel speedup: {residue_time/parallel_time:.2f}x")
    
    def benchmark_accuracy(self):
        """精度検証ベンチマーク"""
        print("\n\n🎯 Accuracy Verification")
        print("-"*40)
        
        # 小規模データで精度を検証
        n_frames = 1000
        n_atoms = 100
        trajectory = self._generate_test_trajectory(n_frames, n_atoms, seed=42)
        
        config = MDConfig()
        
        # GPU結果
        if self.gpu_available:
            detector_gpu = MDLambda3DetectorGPU(config, device='gpu')
            result_gpu = detector_gpu.analyze(trajectory)
            scores_gpu = result_gpu.anomaly_scores['combined']
        
        # CPU結果
        detector_cpu = MDLambda3DetectorGPU(config, device='cpu')
        result_cpu = detector_cpu.analyze(trajectory)
        scores_cpu = result_cpu.anomaly_scores['combined']
        
        if self.gpu_available:
            # 差分解析
            diff = np.abs(scores_gpu - scores_cpu)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")
            print(f"  Relative error: {mean_diff/np.mean(np.abs(scores_cpu)):.2%}")
            
            # 相関
            correlation = np.corrcoef(scores_gpu, scores_cpu)[0, 1]
            print(f"  Correlation: {correlation:.6f}")
            
            # 検出一致率
            threshold = 2.0
            detections_gpu = scores_gpu > threshold
            detections_cpu = scores_cpu > threshold
            agreement = np.mean(detections_gpu == detections_cpu)
            print(f"  Detection agreement: {agreement:.2%}")
    
    # === ヘルパーメソッド ===
    
    def _generate_test_trajectory(self, n_frames: int, n_atoms: int, 
                                seed: Optional[int] = None) -> np.ndarray:
        """テスト用軌道生成"""
        if seed is not None:
            np.random.seed(seed)
        
        # ランダムウォーク的な軌道
        trajectory = np.zeros((n_frames, n_atoms, 3))
        trajectory[0] = np.random.randn(n_atoms, 3) * 10
        
        for i in range(1, n_frames):
            trajectory[i] = trajectory[i-1] + np.random.randn(n_atoms, 3) * 0.1
        
        return trajectory.astype(np.float32)
    
    def _benchmark_gpu(self, trajectory: np.ndarray, config: MDConfig) -> float:
        """GPU版ベンチマーク"""
        try:
            with self._gpu_memory_monitor():
                detector = MDLambda3DetectorGPU(config, device='gpu')
                start = time.time()
                _ = detector.analyze(trajectory)
                return time.time() - start
        except Exception as e:
            print(f"    GPU error: {str(e)}")
            return None
    
    def _benchmark_cpu(self, trajectory: np.ndarray, config: MDConfig) -> float:
        """CPU版ベンチマーク"""
        try:
            detector = MDLambda3DetectorGPU(config, device='cpu')
            start = time.time()
            _ = detector.analyze(trajectory)
            return time.time() - start
        except Exception as e:
            print(f"    CPU error: {str(e)}")
            return None
    
    def _benchmark_md_features(self, trajectory: np.ndarray, use_gpu: bool) -> float:
        """MD特徴抽出のベンチマーク"""
        from ..structures.md_features_gpu import MDFeaturesGPU
        
        extractor = MDFeaturesGPU(force_cpu=not use_gpu)
        config = MDConfig()
        
        start = time.time()
        _ = extractor.extract_md_features(trajectory, config)
        return time.time() - start
    
    def _benchmark_lambda_structures(self, trajectory: np.ndarray, use_gpu: bool) -> float:
        """Lambda構造計算のベンチマーク"""
        from ..structures.lambda_structures_gpu import LambdaStructuresGPU
        from ..structures.md_features_gpu import MDFeaturesGPU
        
        # 前処理
        extractor = MDFeaturesGPU(force_cpu=not use_gpu)
        md_features = extractor.extract_md_features(trajectory, MDConfig())
        
        computer = LambdaStructuresGPU(force_cpu=not use_gpu)
        
        start = time.time()
        _ = computer.compute_lambda_structures(trajectory, md_features, 100)
        return time.time() - start
    
    def _benchmark_anomaly_detection(self, trajectory: np.ndarray, use_gpu: bool) -> float:
        """異常検出のベンチマーク"""
        # 簡略化のため、ダミーデータで測定
        from ..detection.anomaly_detection_gpu import AnomalyDetectorGPU
        
        detector = AnomalyDetectorGPU(force_cpu=not use_gpu)
        series = np.random.randn(trajectory.shape[0])
        
        start = time.time()
        _ = detector.detect_local_anomalies(series, window=50)
        return time.time() - start
    
    def _benchmark_boundary_detection(self, trajectory: np.ndarray, use_gpu: bool) -> float:
        """境界検出のベンチマーク"""
        from ..detection.boundary_detection_gpu import BoundaryDetectorGPU
        
        detector = BoundaryDetectorGPU(force_cpu=not use_gpu)
        
        # ダミー構造
        structures = {
            'rho_T': np.random.randn(trajectory.shape[0]),
            'Q_cumulative': np.cumsum(np.random.randn(trajectory.shape[0]-1))
        }
        
        start = time.time()
        _ = detector.detect_structural_boundaries(structures, 100)
        return time.time() - start
    
    def _benchmark_extended_detection(self, trajectory: np.ndarray, use_gpu: bool) -> float:
        """拡張検出のベンチマーク"""
        from ..detection.extended_detection_gpu import ExtendedDetectorGPU
        
        detector = ExtendedDetectorGPU(force_cpu=not use_gpu)
        
        structures = {
            'rho_T': np.random.randn(trajectory.shape[0]),
            'sigma_s': np.random.rand(trajectory.shape[0])
        }
        
        start = time.time()
        _ = detector.detect_periodic_transitions(structures)
        return time.time() - start
    
    def _get_gpu_utilization(self) -> float:
        """GPU使用率取得"""
        if self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    return gpus[0].load * 100
            except:
                pass
        return 0.0
    
    def _print_result(self, result: BenchmarkResult):
        """結果表示"""
        print(f"\n  Results for {result.test_name}:")
        print(f"    Frames: {result.n_frames}, Atoms: {result.n_atoms}")
        if result.gpu_time > 0:
            print(f"    GPU: {result.gpu_time:.2f}s, CPU: {result.cpu_time:.2f}s")
            print(f"    Speedup: {result.speedup:.1f}x")
            print(f"    Throughput: {result.throughput_fps:.1f} fps")
            print(f"    GPU Memory: {result.gpu_memory_used:.2f} GB")
    
    def _plot_scalability(self, results: List[Dict]):
        """スケーラビリティプロット"""
        df = pd.DataFrame(results)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 実行時間
        ax1.plot(df['n_frames'], df['time'], 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Frames')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Execution Time Scalability')
        ax1.grid(True, alpha=0.3)
        
        # スループット
        ax2.plot(df['n_frames'], df['throughput'], 'o-', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Number of Frames')
        ax2.set_ylabel('Throughput (fps)')
        ax2.set_title('Throughput Scalability')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/scalability.png", dpi=150)
        plt.close()
    
    def _plot_memory_efficiency(self, results: List[Dict]):
        """メモリ効率プロット"""
        success_results = [r for r in results if r.get('success', False)]
        
        if not success_results:
            return
        
        df = pd.DataFrame(success_results)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(df['data_size_gb'], df['gpu_memory_gb'], 'o-', linewidth=2, markersize=8)
        ax.plot([0, df['data_size_gb'].max()], [0, df['data_size_gb'].max()], 
               'k--', alpha=0.5, label='1:1 ratio')
        
        ax.set_xlabel('Data Size (GB)')
        ax.set_ylabel('GPU Memory Used (GB)')
        ax.set_title('Memory Efficiency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/memory_efficiency.png", dpi=150)
        plt.close()
    
    def _plot_component_performance(self, times: Dict[str, float]):
        """コンポーネント性能プロット"""
        components = []
        gpu_times = []
        cpu_times = []
        
        for key, value in times.items():
            if '_GPU' in key:
                components.append(key.replace('_GPU', ''))
                gpu_times.append(value)
            elif '_CPU' in key:
                cpu_times.append(value)
        
        x = np.arange(len(components))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, gpu_times, width, label='GPU', alpha=0.8)
        bars2 = ax.bar(x + width/2, cpu_times, width, label='CPU', alpha=0.8)
        
        ax.set_xlabel('Component')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Component Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # スピードアップ表示
        for i, (gpu_t, cpu_t) in enumerate(zip(gpu_times, cpu_times)):
            if gpu_t > 0:
                speedup = cpu_t / gpu_t
                ax.text(i, max(gpu_t, cpu_t) + 0.1, f'{speedup:.1f}x',
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/component_performance.png", dpi=150)
        plt.close()
    
    def generate_report(self):
        """総合レポート生成"""
        print("\n\n📝 Generating Benchmark Report...")
        
        # 結果をDataFrameに
        df = pd.DataFrame([vars(r) for r in self.results])
        
        # サマリー統計
        summary = {
            'total_tests': len(self.results),
            'avg_speedup': df['speedup'].mean(),
            'max_speedup': df['speedup'].max(),
            'avg_throughput': df['throughput_fps'].mean(),
            'total_gpu_time': df['gpu_time'].sum(),
            'total_cpu_time': df['cpu_time'].sum()
        }
        
        # テキストレポート
        report = f"""
Lambda³ GPU Benchmark Report
{'='*60}

Summary Statistics:
- Total tests run: {summary['total_tests']}
- Average speedup: {summary['avg_speedup']:.1f}x
- Maximum speedup: {summary['max_speedup']:.1f}x
- Average throughput: {summary['avg_throughput']:.1f} fps
- Total GPU time: {summary['total_gpu_time']:.1f}s
- Total CPU time: {summary['total_cpu_time']:.1f}s

Detailed Results:
{df.to_string()}

Environment:
- GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode() if self.gpu_available else 'N/A'}
- CUDA: {cp.cuda.runtime.runtimeGetVersion() if self.gpu_available else 'N/A'}
- CuPy: {cp.__version__ if self.gpu_available else 'N/A'}
- CPU: {psutil.cpu_count()} cores
- RAM: {psutil.virtual_memory().total / 1e9:.1f} GB
"""
        
        # 保存
        with open(f"{self.output_dir}/benchmark_report.txt", 'w') as f:
            f.write(report)
        
        # JSON形式でも保存
        with open(f"{self.output_dir}/benchmark_results.json", 'w') as f:
            json.dump({
                'summary': summary,
                'results': [vars(r) for r in self.results],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        
        print(f"\n✓ Report saved to {self.output_dir}/")
        print(f"  - benchmark_report.txt")
        print(f"  - benchmark_results.json")
        print(f"  - scalability.png")
        print(f"  - memory_efficiency.png")
        print(f"  - component_performance.png")


def run_quick_benchmark():
    """クイックベンチマーク（開発用）"""
    print("🚀 Running quick Lambda³ GPU benchmark...")
    
    suite = Lambda3BenchmarkSuite()
    
    # 小規模テストのみ
    n_frames = 1000
    n_atoms = 100
    trajectory = suite._generate_test_trajectory(n_frames, n_atoms)
    
    config = MDConfig()
    
    # GPU
    if suite.gpu_available:
        start = time.time()
        detector_gpu = MDLambda3DetectorGPU(config, device='gpu')
        result_gpu = detector_gpu.analyze(trajectory)
        gpu_time = time.time() - start
        print(f"\nGPU time: {gpu_time:.3f}s")
    
    # CPU
    start = time.time()
    detector_cpu = MDLambda3DetectorGPU(config, device='cpu')
    result_cpu = detector_cpu.analyze(trajectory)
    cpu_time = time.time() - start
    print(f"CPU time: {cpu_time:.3f}s")
    
    if suite.gpu_available:
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Lambda³ GPU Benchmark Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark only')
    parser.add_argument('--output', default='./benchmark_results', help='Output directory')
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_benchmark()
    else:
        suite = Lambda3BenchmarkSuite(output_dir=args.output)
        suite.run_all_benchmarks()
