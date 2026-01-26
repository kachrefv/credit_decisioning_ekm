"""
Stress testing framework for the Credithos EKM system.
Tests system performance under high load, large datasets, and concurrent operations.
"""
import time
import threading
import numpy as np
import random
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any
import sys
import os
import psutil
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ekm.core.models import Episode, AKU, GKU
from src.ekm.core.engine import EKM
from src.ekm.core.efficient_indexing import ScalableEKM
from src.ekm.benchmarking.performance_benchmarks import PerformanceBenchmarkSuite


@dataclass
class StressTestConfig:
    """Configuration for stress testing."""
    # Dataset size parameters
    min_dataset_size: int = 100
    max_dataset_size: int = 10000
    step_size: int = 1000
    
    # Concurrency parameters
    min_concurrent_threads: int = 1
    max_concurrent_threads: int = 50
    concurrency_step: int = 10
    
    # Operation parameters
    num_queries_per_thread: int = 100
    num_ingestions_per_thread: int = 50
    
    # Time limits
    max_test_duration_seconds: int = 300  # 5 minutes
    warmup_duration_seconds: int = 10
    
    # Memory limits
    max_memory_mb: int = 4096  # 4GB


class StressTester:
    """Main stress testing class."""
    
    def __init__(self, config: StressTestConfig = None):
        self.config = config or StressTestConfig()
        self.results = {}
        self.system_monitor = SystemMonitor()
    
    def generate_synthetic_data(self, size: int, embedding_dim: int = 768) -> List[Episode]:
        """Generate synthetic episodes for testing."""
        episodes = []
        for i in range(size):
            embedding = np.random.randn(embedding_dim).astype(np.float32)
            episode = Episode(
                id=f"synthetic_ep_{i}",
                content=f"Synthetic episode content {i} with diverse information for stress testing",
                embedding=embedding,
                metadata={
                    "timestamp": time.time() - random.randint(0, 86400 * 30),  # Random timestamp in last 30 days
                    "category": random.choice(["personal", "business", "mortgage", "auto"]),
                    "risk_score": random.uniform(0.1, 0.9)
                }
            )
            episodes.append(episode)
        return episodes
    
    def generate_query_embeddings(self, num_queries: int, embedding_dim: int = 768) -> List[np.ndarray]:
        """Generate query embeddings for testing."""
        return [np.random.randn(embedding_dim).astype(np.float32) for _ in range(num_queries)]
    
    def test_scalability_with_dataset_size(self) -> Dict[str, Any]:
        """Test system performance with varying dataset sizes."""
        print("Testing scalability with dataset size...")
        
        sizes = list(range(
            self.config.min_dataset_size, 
            min(self.config.max_dataset_size + 1, 5001),  # Limit for initial test
            self.config.step_size
        ))
        
        results = {
            'dataset_sizes': [],
            'ingestion_times': [],
            'retrieval_times': [],
            'memory_usage_mb': [],
            'throughput_eps_per_sec': []
        }
        
        for size in sizes:
            print(f"Testing with dataset size: {size}")
            
            # Generate synthetic data
            episodes = self.generate_synthetic_data(size)
            
            # Monitor system resources
            initial_memory = self.system_monitor.get_memory_usage_mb()
            
            # Create EKM instance
            ekm = EKM(d=768, k=10, mesh_threshold=max(50, size//10), use_scalable_index=True)
            
            # Measure ingestion time
            start_time = time.time()
            ekm.ingest_episodes(episodes)
            ingestion_time = time.time() - start_time
            
            # Measure retrieval performance
            query_embeddings = self.generate_query_embeddings(10)  # 10 sample queries
            retrieval_start = time.time()
            for query_emb in query_embeddings:
                _ = ekm.retrieve("stress test query", query_emb)
            retrieval_time = (time.time() - retrieval_start) / len(query_embeddings)  # Average per query
            
            # Monitor final memory usage
            final_memory = self.system_monitor.get_memory_usage_mb()
            memory_used = final_memory - initial_memory
            
            # Calculate throughput
            throughput = size / ingestion_time if ingestion_time > 0 else float('inf')
            
            # Store results
            results['dataset_sizes'].append(size)
            results['ingestion_times'].append(ingestion_time)
            results['retrieval_times'].append(retrieval_time)
            results['memory_usage_mb'].append(memory_used)
            results['throughput_eps_per_sec'].append(throughput)
            
            print(f"  Ingestion time: {ingestion_time:.2f}s")
            print(f"  Avg retrieval time: {retrieval_time:.4f}s")
            print(f"  Memory used: {memory_used:.2f}MB")
            print(f"  Throughput: {throughput:.2f} eps/sec")
            
            # Clean up
            del ekm, episodes, query_embeddings
            gc.collect()
        
        self.results['scalability_test'] = results
        return results
    
    def test_concurrent_operations(self) -> Dict[str, Any]:
        """Test system performance under concurrent operations."""
        print("Testing concurrent operations...")
        
        thread_counts = list(range(
            self.config.min_concurrent_threads,
            min(self.config.max_concurrent_threads + 1, 21),  # Limit for initial test
            self.config.concurrency_step
        ))
        
        results = {
            'thread_counts': [],
            'avg_response_times': [],
            'throughput_requests_per_sec': [],
            'error_rates': [],
            'memory_usage_mb': []
        }
        
        for num_threads in thread_counts:
            print(f"Testing with {num_threads} concurrent threads...")
            
            # Generate shared data
            episodes = self.generate_synthetic_data(1000)  # 1000 episodes for all threads
            query_embeddings = self.generate_query_embeddings(50)  # 50 queries for all threads
            
            # Create shared EKM instance
            ekm = EKM(d=768, k=10, mesh_threshold=500, use_scalable_index=True)
            ekm.ingest_episodes(episodes[:500])  # Pre-populate with half the data
            
            # Shared counters for thread-safe operations
            import threading
            response_times = []
            error_count = 0
            error_lock = threading.Lock()
            
            def worker_task(worker_id: int):
                nonlocal error_count
                local_response_times = []
                
                # Each worker performs ingestion and retrieval
                for i in range(self.config.num_ingestions_per_thread // num_threads):
                    try:
                        # Create a few episodes to ingest
                        worker_episodes = [
                            Episode(
                                id=f"worker_{worker_id}_ep_{i}_{j}",
                                content=f"Worker {worker_id} episode {i}-{j}",
                                embedding=np.random.randn(768).astype(np.float32),
                                metadata={"timestamp": time.time()}
                            )
                            for j in range(2)  # 2 episodes per iteration
                        ]
                        
                        start_time = time.time()
                        ekm.ingest_episodes(worker_episodes)
                        local_response_times.append(time.time() - start_time)
                    except Exception as e:
                        with error_lock:
                            error_count += 1
                        print(f"Error in worker {worker_id} during ingestion: {e}")
                
                # Perform retrieval operations
                for i in range(self.config.num_queries_per_thread // num_threads):
                    try:
                        query_idx = random.randint(0, len(query_embeddings) - 1)
                        query_emb = query_embeddings[query_idx]
                        
                        start_time = time.time()
                        _ = ekm.retrieve(f"query from worker {worker_id}", query_emb)
                        local_response_times.append(time.time() - start_time)
                    except Exception as e:
                        with error_lock:
                            error_count += 1
                        print(f"Error in worker {worker_id} during retrieval: {e}")
                
                # Add local results to shared list
                response_times.extend(local_response_times)
            
            # Start concurrent workers
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker_task, i) for i in range(num_threads)]
                for future in as_completed(futures):
                    try:
                        future.result()  # Wait for all tasks to complete
                    except Exception as e:
                        print(f"Worker task failed: {e}")
            
            total_time = time.time() - start_time
            total_operations = len(response_times)
            
            avg_response_time = np.mean(response_times) if response_times else 0
            total_requests = (self.config.num_ingestions_per_thread // num_threads) * num_threads + \
                           (self.config.num_queries_per_thread // num_threads) * num_threads
            throughput = total_requests / total_time if total_time > 0 else 0
            error_rate = error_count / total_requests if total_requests > 0 else 0
            memory_usage = self.system_monitor.get_memory_usage_mb()
            
            # Store results
            results['thread_counts'].append(num_threads)
            results['avg_response_times'].append(avg_response_time)
            results['throughput_requests_per_sec'].append(throughput)
            results['error_rates'].append(error_rate)
            results['memory_usage_mb'].append(memory_usage)
            
            print(f"  Avg response time: {avg_response_time:.4f}s")
            print(f"  Throughput: {throughput:.2f} reqs/sec")
            print(f"  Error rate: {error_rate:.4f}")
            print(f"  Memory usage: {memory_usage:.2f}MB")
        
        self.results['concurrency_test'] = results
        return results
    
    def test_memory_stress(self) -> Dict[str, Any]:
        """Test memory usage under stress conditions."""
        print("Testing memory stress...")
        
        # Gradually increase data size until memory limit or time limit
        current_size = self.config.min_dataset_size
        memory_limit_reached = False
        time_limit_reached = False
        
        results = {
            'dataset_sizes': [],
            'memory_usage_mb': [],
            'memory_percent': [],
            'ingestion_times': []
        }
        
        start_test_time = time.time()
        
        while not memory_limit_reached and not time_limit_reached:
            if current_size > self.config.max_dataset_size:
                break
                
            if time.time() - start_test_time > self.config.max_test_duration_seconds:
                time_limit_reached = True
                break
            
            print(f"Testing memory usage with dataset size: {current_size}")
            
            # Generate data
            episodes = self.generate_synthetic_data(current_size)
            
            # Monitor memory before
            memory_before = self.system_monitor.get_memory_usage_mb()
            percent_before = self.system_monitor.get_memory_percent()
            
            # Create and populate EKM
            ekm = EKM(d=768, k=10, mesh_threshold=max(50, current_size//10), use_scalable_index=True)
            
            start_ingest = time.time()
            ekm.ingest_episodes(episodes)
            ingestion_time = time.time() - start_ingest
            
            # Monitor memory after
            memory_after = self.system_monitor.get_memory_usage_mb()
            memory_used = memory_after - memory_before
            memory_percent = self.system_monitor.get_memory_percent()
            
            # Check if we're approaching limits
            memory_limit_reached = memory_after > self.config.max_memory_mb
            
            # Store results
            results['dataset_sizes'].append(current_size)
            results['memory_usage_mb'].append(memory_used)
            results['memory_percent'].append(memory_percent)
            results['ingestion_times'].append(ingestion_time)
            
            print(f"  Memory used: {memory_used:.2f}MB ({memory_percent:.2f}%)")
            print(f"  Ingestion time: {ingestion_time:.2f}s")
            
            # Increase size for next iteration
            current_size = min(current_size * 2, current_size + 1000)
            
            # Clean up
            del ekm, episodes
            gc.collect()
        
        self.results['memory_stress_test'] = results
        return results
    
    def test_longevity_stress(self, duration_minutes: int = 10) -> Dict[str, Any]:
        """Test system performance over extended period."""
        print(f"Testing longevity stress for {duration_minutes} minutes...")
        
        end_time = time.time() + (duration_minutes * 60)
        
        results = {
            'timestamps': [],
            'memory_usage_mb': [],
            'response_times': [],
            'operations_count': [],
            'error_count': []
        }
        
        # Create EKM instance
        ekm = EKM(d=768, k=10, mesh_threshold=100, use_scalable_index=True)
        
        # Pre-populate with some data
        initial_episodes = self.generate_synthetic_data(500)
        ekm.ingest_episodes(initial_episodes)
        
        operation_count = 0
        error_count = 0
        query_embeddings = self.generate_query_embeddings(100)
        
        while time.time() < end_time:
            current_time = time.time()
            
            try:
                # Perform random operations
                op_type = random.choice(['ingest', 'retrieve', 'consolidate'])
                
                if op_type == 'ingest':
                    # Ingest new episodes
                    new_episodes = [
                        Episode(
                            id=f"longevity_ep_{operation_count}_{i}",
                            content=f"Longevity test episode {operation_count}-{i}",
                            embedding=np.random.randn(768).astype(np.float32),
                            metadata={"timestamp": current_time}
                        )
                        for i in range(random.randint(1, 5))
                    ]
                    ekm.ingest_episodes(new_episodes)
                    
                elif op_type == 'retrieve':
                    # Perform retrieval
                    query_idx = random.randint(0, len(query_embeddings) - 1)
                    query_emb = query_embeddings[query_idx]
                    
                    start_time = time.time()
                    _ = ekm.retrieve("longevity query", query_emb)
                    response_time = time.time() - start_time
                    
                    results['response_times'].append(response_time)
                
                elif op_type == 'consolidate' and operation_count % 50 == 0:
                    # Occasionally run consolidation
                    ekm.consolidate()
                
                operation_count += 1
                
            except Exception as e:
                error_count += 1
                print(f"Error during longevity test: {e}")
            
            # Log metrics periodically
            if operation_count % 10 == 0:
                results['timestamps'].append(current_time)
                results['memory_usage_mb'].append(self.system_monitor.get_memory_usage_mb())
                results['operations_count'].append(operation_count)
                results['error_count'].append(error_count)
        
        results['final_error_count'] = error_count
        results['total_operations'] = operation_count
        
        self.results['longevity_test'] = results
        return results
    
    def run_all_stress_tests(self) -> Dict[str, Any]:
        """Run all stress tests and return comprehensive results."""
        print("Starting comprehensive stress testing...")
        
        # Run each test
        try:
            self.test_scalability_with_dataset_size()
        except Exception as e:
            print(f"Scalability test failed: {e}")
        
        try:
            self.test_concurrent_operations()
        except Exception as e:
            print(f"Concurrency test failed: {e}")
        
        try:
            self.test_memory_stress()
        except Exception as e:
            print(f"Memory stress test failed: {e}")
        
        try:
            self.test_longevity_stress(duration_minutes=2)  # Shorter for initial test
        except Exception as e:
            print(f"Longevity test failed: {e}")
        
        return self.results
    
    def generate_stress_report(self) -> str:
        """Generate a comprehensive stress test report."""
        report = []
        report.append("# Credithos EKM Stress Testing Report")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if 'scalability_test' in self.results:
            st = self.results['scalability_test']
            report.append("## Scalability Test Results")
            report.append(f"- Max dataset size tested: {max(st['dataset_sizes']) if st['dataset_sizes'] else 0}")
            report.append(f"- Max ingestion throughput: {max(st['throughput_eps_per_sec']) if st['throughput_eps_per_sec'] else 0:.2f} eps/sec")
            report.append(f"- Min avg retrieval time: {min(st['retrieval_times']) if st['retrieval_times'] else 0:.4f}s")
            report.append(f"- Max memory usage: {max(st['memory_usage_mb']) if st['memory_usage_mb'] else 0:.2f}MB")
            report.append("")
        
        if 'concurrency_test' in self.results:
            ct = self.results['concurrency_test']
            report.append("## Concurrency Test Results")
            report.append(f"- Max concurrent threads tested: {max(ct['thread_counts']) if ct['thread_counts'] else 0}")
            report.append(f"- Max throughput under load: {max(ct['throughput_requests_per_sec']) if ct['throughput_requests_per_sec'] else 0:.2f} reqs/sec")
            report.append(f"- Min avg response time: {min(ct['avg_response_times']) if ct['avg_response_times'] else 0:.4f}s")
            report.append(f"- Max error rate: {max(ct['error_rates']) if ct['error_rates'] else 0:.4f}")
            report.append("")
        
        if 'memory_stress_test' in self.results:
            mt = self.results['memory_stress_test']
            report.append("## Memory Stress Test Results")
            report.append(f"- Max dataset size before limits: {max(mt['dataset_sizes']) if mt['dataset_sizes'] else 0}")
            report.append(f"- Peak memory usage: {max(mt['memory_usage_mb']) if mt['memory_usage_mb'] else 0:.2f}MB")
            report.append(f"- Peak memory percentage: {max(mt['memory_percent']) if mt['memory_percent'] else 0:.2f}%")
            report.append("")
        
        if 'longevity_test' in self.results:
            lt = self.results['longevity_test']
            report.append("## Longevity Test Results")
            report.append(f"- Total operations performed: {lt['total_operations'] if 'total_operations' in lt else 0}")
            report.append(f"- Total errors during test: {lt['final_error_count'] if 'final_error_count' in lt else 0}")
            report.append(f"- Average response time: {np.mean(lt['response_times']):.4f}s" if lt['response_times'] else "- No response time data")
            report.append("")
        
        return "\n".join(report)
    
    def plot_stress_results(self):
        """Plot stress test results."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Credithos EKM Stress Test Results', fontsize=16)
            
            # Plot 1: Scalability - Dataset size vs Ingestion time
            if 'scalability_test' in self.results:
                st = self.results['scalability_test']
                axes[0, 0].plot(st['dataset_sizes'], st['ingestion_times'], 'b-o')
                axes[0, 0].set_title('Dataset Size vs Ingestion Time')
                axes[0, 0].set_xlabel('Dataset Size')
                axes[0, 0].set_ylabel('Ingestion Time (s)')
                axes[0, 0].grid(True)
            
            # Plot 2: Concurrency - Thread count vs Response time
            if 'concurrency_test' in self.results:
                ct = self.results['concurrency_test']
                axes[0, 1].plot(ct['thread_counts'], ct['avg_response_times'], 'r-o')
                axes[0, 1].set_title('Concurrency vs Response Time')
                axes[0, 1].set_xlabel('Thread Count')
                axes[0, 1].set_ylabel('Avg Response Time (s)')
                axes[0, 1].grid(True)
            
            # Plot 3: Memory usage over time (from longevity test)
            if 'longevity_test' in self.results:
                lt = self.results['longevity_test']
                if lt['timestamps'] and lt['memory_usage_mb']:
                    axes[1, 0].plot(lt['timestamps'], lt['memory_usage_mb'], 'g-')
                    axes[1, 0].set_title('Memory Usage Over Time (Longevity)')
                    axes[1, 0].set_xlabel('Time')
                    axes[1, 0].set_ylabel('Memory Usage (MB)')
                    axes[1, 0].grid(True)
            
            # Plot 4: Scalability - Dataset size vs Memory usage
            if 'scalability_test' in self.results:
                st = self.results['scalability_test']
                axes[1, 1].plot(st['dataset_sizes'], st['memory_usage_mb'], 'm-o')
                axes[1, 1].set_title('Dataset Size vs Memory Usage')
                axes[1, 1].set_xlabel('Dataset Size')
                axes[1, 1].set_ylabel('Memory Usage (MB)')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig('stress_test_results.png', dpi=300, bbox_inches='tight')
            print("Stress test plots saved as 'stress_test_results.png'")
            
        except ImportError:
            print("Matplotlib not available, skipping plot generation")


class SystemMonitor:
    """System resource monitoring utility."""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            # Get memory info for the current process
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert bytes to MB
        except:
            # Fallback to overall system memory if process info unavailable
            return psutil.virtual_memory().used / 1024 / 1024
    
    def get_memory_percent(self) -> float:
        """Get current memory usage as percentage."""
        try:
            return self.process.memory_percent()
        except:
            # Fallback to overall system memory percent
            return psutil.virtual_memory().percent
    
    def get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent()


def run_stress_tests():
    """Run the complete stress testing suite."""
    print("Starting Credithos EKM Stress Testing Suite...")
    
    # Create stress tester with conservative settings for initial run
    config = StressTestConfig(
        min_dataset_size=100,
        max_dataset_size=2000,  # Reduced for initial test
        step_size=500,
        max_concurrent_threads=10,  # Reduced for initial test
        max_test_duration_seconds=120,  # 2 minutes
        max_memory_mb=2048  # 2GB
    )
    
    stress_tester = StressTester(config)
    
    # Run all stress tests
    results = stress_tester.run_all_stress_tests()
    
    # Generate report
    report = stress_tester.generate_stress_report()
    print("\n" + "="*60)
    print("STRESS TESTING COMPLETE")
    print("="*60)
    print(report)
    
    # Generate plots
    stress_tester.plot_stress_results()
    
    print("\nStress testing completed successfully!")
    return stress_tester


if __name__ == "__main__":
    stress_tester = run_stress_tests()