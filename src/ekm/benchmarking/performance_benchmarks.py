"""
Performance benchmarking module for Credithos EKM system.
Validates the performance claims made in the technical report.
"""
import time
import numpy as np
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from ..core.engine import EKM
from ..core.models import Episode, AKU
from ..core.retrieval import EKMRetriever
from ..core.graph import GraphEngine


class PerformanceBenchmarkSuite:
    """
    Comprehensive benchmarking suite to validate performance claims from the technical report.
    """
    
    def __init__(self):
        self.results = {}
        
    def benchmark_retrieval_precision(self, ekm_system: EKM, test_queries: List[np.ndarray], 
                                    ground_truth: List[List[int]], k: int = 10) -> Dict[str, Any]:
        """
        Benchmark retrieval precision against ground truth.
        
        Args:
            ekm_system: The EKM system to benchmark
            test_queries: List of query embeddings
            ground_truth: List of ground truth indices for each query
            k: Number of top results to consider for precision calculation
            
        Returns:
            Dictionary with precision metrics
        """
        start_time = time.time()
        
        precisions = []
        for i, query in enumerate(test_queries):
            retrieved_results = ekm_system.retrieve("test query", query)
            retrieved_indices = [idx for idx, _ in retrieved_results[:k]]
            
            # Calculate precision at k
            relevant_retrieved = len(set(retrieved_indices) & set(ground_truth[i]))
            precision_at_k = relevant_retrieved / min(k, len(ground_truth[i]))
            precisions.append(precision_at_k)
        
        avg_precision = np.mean(precisions)
        std_precision = np.std(precisions)
        
        elapsed_time = time.time() - start_time
        
        result = {
            'precision_at_k': avg_precision,
            'std_precision': std_precision,
            'num_queries': len(test_queries),
            'time_elapsed': elapsed_time,
            'relative_improvement_vs_baseline': None  # Will calculate later
        }
        
        self.results['retrieval_precision'] = result
        return result
    
    def benchmark_latency(self, ekm_system: EKM, test_queries: List[np.ndarray], 
                         num_iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark system latency for retrieval operations.
        
        Args:
            ekm_system: The EKM system to benchmark
            test_queries: List of query embeddings
            num_iterations: Number of iterations to average over
            
        Returns:
            Dictionary with latency metrics
        """
        latencies = []
        
        for _ in range(num_iterations):
            query = np.random.randn(ekm_system.d)  # Use system's embedding dimension
            start_time = time.time()
            
            # Perform a retrieval operation
            _ = ekm_system.retrieve("test query", query)
            
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        result = {
            'avg_latency_ms': avg_latency,
            'median_latency_ms': median_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'num_iterations': num_iterations
        }
        
        self.results['latency'] = result
        return result
    
    def benchmark_scalability(self, sizes: List[int]) -> Dict[str, Any]:
        """
        Benchmark system scalability with different data sizes.
        
        Args:
            sizes: List of data sizes to test
            
        Returns:
            Dictionary with scalability metrics
        """
        scalability_results = {
            'sizes': sizes,
            'construction_times': [],
            'memory_usage_mb': [],
            'retrieval_times': []
        }
        
        for size in sizes:
            # Create synthetic data
            episodes = []
            for i in range(size):
                embedding = np.random.randn(768)
                episode = Episode(
                    id=f"ep_{i}",
                    content=f"Synthetic episode {i}",
                    embedding=embedding,
                    metadata={"timestamp": time.time()}
                )
                episodes.append(episode)
            
            # Measure construction time
            start_time = time.time()
            ekm = EKM(d=768, k=10, mesh_threshold=size//2)  # Adjust threshold appropriately
            ekm.ingest_episodes(episodes)
            construction_time = time.time() - start_time
            
            # Measure retrieval time
            query_embedding = np.random.randn(768)
            retrieval_start = time.time()
            _ = ekm.retrieve("test", query_embedding)
            retrieval_time = (time.time() - retrieval_start) * 1000  # ms
            
            scalability_results['construction_times'].append(construction_time)
            scalability_results['retrieval_times'].append(retrieval_time)
            
            # Estimate memory usage (approximate)
            import sys
            memory_mb = sys.getsizeof(ekm) / (1024 * 1024)
            scalability_results['memory_usage_mb'].append(memory_mb)
        
        self.results['scalability'] = scalability_results
        return scalability_results
    
    def benchmark_memory_efficiency(self, ekm_system: EKM, baseline_size: int = 1000) -> Dict[str, Any]:
        """
        Benchmark memory efficiency of consolidation protocol.
        
        Args:
            ekm_system: The EKM system to benchmark
            baseline_size: Baseline number of nodes for comparison
            
        Returns:
            Dictionary with memory efficiency metrics
        """
        import psutil
        import os
        
        # Get initial memory usage
        initial_process = psutil.Process(os.getpid())
        initial_memory = initial_process.memory_info().rss / 1024 / 1024  # MB
        
        # Create baseline data
        episodes = []
        for i in range(baseline_size):
            embedding = np.random.randn(768)
            episode = Episode(
                id=f"ep_{i}",
                content=f"Synthetic episode {i}",
                embedding=embedding,
                metadata={"timestamp": time.time()}
            )
            episodes.append(episode)
        
        ekm_system.ingest_episodes(episodes)
        
        # Get memory after ingestion
        after_ingestion_memory = initial_process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform consolidation
        start_consolidation = time.time()
        ekm_system.consolidate()
        consolidation_time = time.time() - start_consolidation
        
        # Get memory after consolidation
        after_consolidation_memory = initial_process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate memory reduction
        initial_nodes = baseline_size
        final_nodes = len(ekm_system.akus)
        memory_reduction_percentage = ((initial_memory - after_consolidation_memory) / initial_memory) * 100 if initial_memory > 0 else 0
        
        result = {
            'initial_nodes': initial_nodes,
            'final_nodes': final_nodes,
            'node_reduction_percentage': ((initial_nodes - final_nodes) / initial_nodes) * 100 if initial_nodes > 0 else 0,
            'initial_memory_mb': initial_memory,
            'after_ingestion_memory_mb': after_ingestion_memory,
            'after_consolidation_memory_mb': after_consolidation_memory,
            'memory_reduction_percentage': memory_reduction_percentage,
            'consolidation_time_seconds': consolidation_time
        }
        
        self.results['memory_efficiency'] = result
        return result
    
    def compare_with_baseline_rag(self, ekm_system: EKM, queries: List[np.ndarray], 
                                ground_truth: List[List[int]], k: int = 10) -> Dict[str, Any]:
        """
        Compare EKM performance with baseline RAG system.
        
        Args:
            ekm_system: The EKM system to compare
            queries: List of query embeddings
            ground_truth: Ground truth for each query
            k: Number of top results to consider
            
        Returns:
            Dictionary comparing EKM vs baseline RAG
        """
        # Create a simple baseline RAG system for comparison
        baseline_results = []
        ekm_results = []
        
        for i, query in enumerate(queries):
            # Baseline: Simple cosine similarity search
            all_embeddings = np.stack([aku.embedding for aku in ekm_system.akus])
            similarities = cosine_similarity([query], all_embeddings)[0]
            baseline_top_k_indices = np.argsort(similarities)[-k:][::-1]
            
            # EKM: Use the actual system
            ekm_retrieved = ekm_system.retrieve("test", query)
            ekm_top_k_indices = [ekm_system.akus.index(result[0]) for result in ekm_retrieved[:k]]
            
            # Calculate precision for both
            baseline_relevant = len(set(baseline_top_k_indices) & set(ground_truth[i]))
            baseline_precision = baseline_relevant / min(k, len(ground_truth[i]))
            
            ekm_relevant = len(set(ekm_top_k_indices) & set(ground_truth[i]))
            ekm_precision = ekm_relevant / min(k, len(ground_truth[i]))
            
            baseline_results.append(baseline_precision)
            ekm_results.append(ekm_precision)
        
        baseline_avg_precision = np.mean(baseline_results)
        ekm_avg_precision = np.mean(ekm_results)
        
        improvement_percentage = ((ekm_avg_precision - baseline_avg_precision) / baseline_avg_precision) * 100 if baseline_avg_precision > 0 else 0
        
        result = {
            'baseline_precision': baseline_avg_precision,
            'ekm_precision': ekm_avg_precision,
            'improvement_percentage': improvement_percentage,
            'num_queries': len(queries)
        }
        
        self.results['baseline_comparison'] = result
        return result
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive benchmarking report.
        """
        report = []
        report.append("# Credithos EKM Performance Benchmarking Report\n")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if 'retrieval_precision' in self.results:
            rp = self.results['retrieval_precision']
            report.append("## Retrieval Precision")
            report.append(f"- Average Precision@{10}: {rp['precision_at_k']:.4f}")
            report.append(f"- Standard Deviation: {rp['std_precision']:.4f}")
            report.append(f"- Total Queries: {rp['num_queries']}")
            report.append(f"- Time Elapsed: {rp['time_elapsed']:.2f}s\n")
        
        if 'latency' in self.results:
            lat = self.results['latency']
            report.append("## Latency Performance")
            report.append(f"- Average Latency: {lat['avg_latency_ms']:.2f}ms")
            report.append(f"- Median Latency: {lat['median_latency_ms']:.2f}ms")
            report.append(f"- 95th Percentile: {lat['p95_latency_ms']:.2f}ms")
            report.append(f"- 99th Percentile: {lat['p99_latency_ms']:.2f}ms")
            report.append(f"- Target (<200ms): {'✓' if lat['avg_latency_ms'] < 200 else '✗'}\n")
        
        if 'scalability' in self.results:
            scal = self.results['scalability']
            report.append("## Scalability Analysis")
            for i, size in enumerate(scal['sizes']):
                report.append(f"- Size {size}: Construction {scal['construction_times'][i]:.2f}s, "
                            f"Retrieval {scal['retrieval_times'][i]:.2f}ms, "
                            f"Memory {scal['memory_usage_mb'][i]:.2f}MB")
            report.append("")
        
        if 'memory_efficiency' in self.results:
            mem = self.results['memory_efficiency']
            report.append("## Memory Efficiency")
            report.append(f"- Initial Nodes: {mem['initial_nodes']}")
            report.append(f"- Final Nodes: {mem['final_nodes']}")
            report.append(f"- Node Reduction: {mem['node_reduction_percentage']:.2f}%")
            report.append(f"- Memory Reduction: {mem['memory_reduction_percentage']:.2f}%")
            report.append(f"- Consolidation Time: {mem['consolidation_time_seconds']:.2f}s")
            report.append(f"- Target (15-30% reduction): {'✓' if 15 <= mem['memory_reduction_percentage'] <= 30 else '✗'}\n")
        
        if 'baseline_comparison' in self.results:
            comp = self.results['baseline_comparison']
            report.append("## Baseline Comparison")
            report.append(f"- Baseline RAG Precision: {comp['baseline_precision']:.4f}")
            report.append(f"- EKM Precision: {comp['ekm_precision']:.4f}")
            report.append(f"- Improvement: {comp['improvement_percentage']:.2f}%")
            report.append(f"- Target (32% improvement): {'✓' if comp['improvement_percentage'] >= 32 else '✗'}\n")
        
        return "\n".join(report)
    
    def plot_scalability_analysis(self):
        """
        Plot scalability analysis if the data exists.
        """
        if 'scalability' not in self.results:
            print("No scalability data available to plot.")
            return
        
        scal = self.results['scalability']
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot construction time
        ax1.plot(scal['sizes'], scal['construction_times'], marker='o')
        ax1.set_title('Construction Time vs Data Size')
        ax1.set_xlabel('Number of Episodes')
        ax1.set_ylabel('Time (seconds)')
        ax1.grid(True)
        
        # Plot retrieval time
        ax2.plot(scal['sizes'], scal['retrieval_times'], marker='s', color='orange')
        ax2.set_title('Retrieval Time vs Data Size')
        ax2.set_xlabel('Number of Episodes')
        ax2.set_ylabel('Time (milliseconds)')
        ax2.grid(True)
        
        # Plot memory usage
        ax3.plot(scal['sizes'], scal['memory_usage_mb'], marker='^', color='green')
        ax3.set_title('Memory Usage vs Data Size')
        ax3.set_xlabel('Number of Episodes')
        ax3.set_ylabel('Memory (MB)')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig('scalability_analysis.png', dpi=300, bbox_inches='tight')
        print("Scalability analysis plot saved as 'scalability_analysis.png'")


def run_comprehensive_benchmark():
    """
    Run a comprehensive benchmark of the EKM system.
    """
    print("Starting comprehensive benchmark of Credithos EKM system...")
    
    # Initialize benchmark suite
    benchmark_suite = PerformanceBenchmarkSuite()
    
    # Create a test EKM system
    ekm = EKM(d=768, k=10, mesh_threshold=50)
    
    # Generate synthetic test data
    print("Generating synthetic test data...")
    test_episodes = []
    for i in range(100):
        embedding = np.random.randn(768)
        episode = Episode(
            id=f"ep_{i}",
            content=f"Synthetic episode {i}",
            embedding=embedding,
            metadata={"timestamp": time.time()}
        )
        test_episodes.append(episode)
    
    # Ingest data into EKM
    print("Ingesting test data into EKM...")
    ekm.ingest_episodes(test_episodes)
    
    # Generate test queries
    test_queries = [np.random.randn(768) for _ in range(20)]
    
    # Create mock ground truth (first 10 episodes are relevant to each query)
    ground_truth = [list(range(10)) for _ in range(20)]
    
    # Run benchmarks
    print("Running retrieval precision benchmark...")
    benchmark_suite.benchmark_retrieval_precision(ekm, test_queries, ground_truth)
    
    print("Running latency benchmark...")
    benchmark_suite.benchmark_latency(ekm, test_queries)
    
    print("Running scalability benchmark...")
    benchmark_suite.benchmark_scalability([100, 200, 500])
    
    print("Running memory efficiency benchmark...")
    benchmark_suite.benchmark_memory_efficiency(ekm, baseline_size=500)
    
    print("Running baseline comparison...")
    benchmark_suite.compare_with_baseline_rag(ekm, test_queries, ground_truth)
    
    # Generate and print report
    report = benchmark_suite.generate_report()
    print("\n" + "="*60)
    print("BENCHMARKING REPORT")
    print("="*60)
    print(report)
    
    # Plot scalability analysis
    benchmark_suite.plot_scalability_analysis()
    
    print("Comprehensive benchmarking complete!")
    return benchmark_suite


if __name__ == "__main__":
    run_comprehensive_benchmark()