import os
import argparse
from typing import Optional
from dask_cuda import LocalCUDACluster
from distributed import Client

class DaskClusterManager:
    def __init__(
        self,
        cuda_visible_devices: str = "0,1",
        rmm_pool_size: float = 0.5,
        enable_cudf_spill: bool = True,
        local_directory: str = "/tmp/dask-scratch",
        use_multi_gpu: bool = True,
        device_memory_limit: str = "auto",
        threads_per_worker: int = 1,
        processes: bool = True,
    ):
        self.cuda_visible_devices = cuda_visible_devices
        self.rmm_pool_size = rmm_pool_size
        self.enable_cudf_spill = enable_cudf_spill
        self.local_directory = local_directory
        self.use_multi_gpu = use_multi_gpu
        self.device_memory_limit = device_memory_limit
        self.threads_per_worker = threads_per_worker
        self.processes = processes
        
        self.cuda_devices = cuda_visible_devices.split(',')
        self.cluster = None
        self.client = None
    
    def start_cluster(self) -> str:
        """Start the Dask cluster and return scheduler address"""
        os.makedirs(self.local_directory, exist_ok=True)
        
        if self.use_multi_gpu:
            print(f"Starting multi-GPU cluster with devices: {self.cuda_visible_devices}")
            self.cluster = self._create_multi_gpu_cluster()
        else:
            print("Starting single-GPU cluster")
            self.cluster = self._create_single_gpu_cluster()
        
        self.client = Client(self.cluster)
        
        # Print cluster information
        print(f"Cluster started successfully!")
        print(f"Dashboard available at: {self.client.dashboard_link}")
        print(f"Scheduler address: {self.client.scheduler.address}")
        
        workers = self.client.scheduler_info()['workers']
        print(f"Number of workers: {len(workers)}")
        
        # Show which GPU each worker is using
        for worker_id, worker_info in workers.items():
            resources = worker_info.get('resources', {})
            print(f"Worker {worker_id}: {resources}")
        
        return self.client.scheduler.address
    
    def _create_multi_gpu_cluster(self) -> LocalCUDACluster:
        """Create multi-GPU cluster where each worker gets its own GPU"""
        return LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=self.cuda_visible_devices,
            rmm_pool_size=self.rmm_pool_size,
            enable_cudf_spill=self.enable_cudf_spill,
            local_directory=self.local_directory,
            processes=self.processes,
            threads_per_worker=self.threads_per_worker,
            device_memory_limit=self.device_memory_limit,
        )
    
    def _create_single_gpu_cluster(self) -> LocalCUDACluster:
        """Create single-GPU cluster"""
        return LocalCUDACluster(
            CUDA_VISIBLE_DEVICES="0",
            rmm_pool_size=self.rmm_pool_size,
            enable_cudf_spill=self.enable_cudf_spill,
            local_directory=self.local_directory,
            processes=self.processes,
            threads_per_worker=self.threads_per_worker,
            device_memory_limit=self.device_memory_limit,
        )
    
    def get_client(self) -> Optional[Client]:
        """Get the current client"""
        return self.client
    
    def get_cluster_info(self) -> dict:
        """Get cluster information"""
        if not self.client:
            return {}
        
        workers = self.client.scheduler_info()['workers']
        return {
            'num_workers': len(workers),
            'cuda_devices': self.cuda_devices,
            'dashboard_link': self.client.dashboard_link,
            'scheduler_address': self.client.scheduler.address,
            'workers': workers
        }
    
    def close(self):
        """Close the cluster and client"""
        if self.client:
            self.client.close()
            print("Dask client closed.")
        
        if self.cluster:
            self.cluster.close()
            print("Dask cluster closed.")


def main():
    parser = argparse.ArgumentParser(description="Dask Cluster Manager")
    
    # Cluster configuration arguments
    parser.add_argument("--cuda-devices", default="0,1", help="CUDA devices to use")
    parser.add_argument("--rmm-pool-size", type=float, default=0.5, help="RMM pool size")
    parser.add_argument("--enable-cudf-spill", action="store_true", default=True, help="Enable cuDF spilling")
    parser.add_argument("--local-directory", default="/tmp/dask-scratch", help="Local directory for spilling")
    parser.add_argument("--no-multi-gpu", action="store_true", help="Use single GPU")
    parser.add_argument("--device-memory-limit", default="auto", help="Device memory limit")
    parser.add_argument("--threads-per-worker", type=int, default=1, help="Threads per worker")
    parser.add_argument("--processes", action="store_true", default=True, help="Use processes instead of threads")
    
    # Add option to just return scheduler address
    parser.add_argument("--return-address", action="store_true", help="Return scheduler address and exit")
    
    args = parser.parse_args()
    
    # Create and start cluster
    cluster_manager = DaskClusterManager(
        cuda_visible_devices=args.cuda_devices,
        rmm_pool_size=args.rmm_pool_size,
        enable_cudf_spill=args.enable_cudf_spill,
        local_directory=args.local_directory,
        use_multi_gpu=not args.no_multi_gpu,
        device_memory_limit=args.device_memory_limit,
        threads_per_worker=args.threads_per_worker,
        processes=args.processes,
    )
    
    scheduler_address = cluster_manager.start_cluster()
    
    if args.return_address:
        print(scheduler_address)
        cluster_manager.close()
        return scheduler_address
    
    # Keep the cluster running
    print("\nCluster is running. Press Ctrl+C to stop...")
    print(f"Scheduler address: {scheduler_address}")
    
    try:
        # Wait for user interrupt
        import signal
        import time
        
        def signal_handler(signum, frame):
            print("\nShutting down cluster...")
            cluster_manager.close()
            exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down cluster...")
        cluster_manager.close()


if __name__ == "__main__":
    main()