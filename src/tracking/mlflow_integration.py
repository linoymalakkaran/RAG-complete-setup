"""
MLflow Integration for RAG System

This module integrates MLflow for:
- Experiment tracking
- Parameter logging
- Metric tracking
- Model versioning
- Artifact storage
- Run comparison
"""

import os
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Try to import MLflow
try:
    import mlflow
    from mlflow import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Run: pip install mlflow")


@dataclass
class ExperimentConfig:
    """Configuration for MLflow experiment."""
    experiment_name: str = "RAG_Experiments"
    tracking_uri: Optional[str] = None
    artifact_location: Optional[str] = None
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class MLflowTracker:
    """
    MLflow integration for RAG system tracking.
    
    Features:
    - Automatic experiment creation
    - Parameter and metric logging
    - Artifact storage
    - Run management
    - Model registry integration
    """
    
    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        auto_log: bool = True
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            config: Experiment configuration
            auto_log: Enable automatic logging
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not installed. Run: pip install mlflow")
        
        self.config = config or ExperimentConfig()
        self.auto_log = auto_log
        self.client = MlflowClient()
        
        # Set tracking URI
        if self.config.tracking_uri:
            mlflow.set_tracking_uri(self.config.tracking_uri)
        
        # Create or get experiment
        self._setup_experiment()
        
        # Current run info
        self.current_run = None
        
        logger.info(f"MLflow tracker initialized for experiment: {self.config.experiment_name}")
    
    def _setup_experiment(self):
        """Create or get experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            
            if experiment is None:
                # Create new experiment
                experiment_id = mlflow.create_experiment(
                    name=self.config.experiment_name,
                    artifact_location=self.config.artifact_location,
                    tags=self.config.tags
                )
                logger.info(f"Created new experiment: {self.config.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.config.experiment_name}")
            
            self.experiment_id = experiment_id
            mlflow.set_experiment(experiment_id=experiment_id)
            
        except Exception as e:
            logger.error(f"Error setting up experiment: {e}")
            raise
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Additional tags
            description: Run description
            
        Returns:
            Active MLflow run
        """
        # End previous run if exists
        if self.current_run:
            self.end_run()
        
        # Prepare tags
        run_tags = tags or {}
        if description:
            run_tags["mlflow.note.content"] = description
        
        # Start run
        self.current_run = mlflow.start_run(
            run_name=run_name,
            tags=run_tags
        )
        
        logger.info(f"Started MLflow run: {self.current_run.info.run_id}")
        return self.current_run
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if self.current_run:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run: {self.current_run.info.run_id}")
            self.current_run = None
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to current run.
        
        Args:
            params: Dictionary of parameters
        """
        if not self.current_run:
            logger.warning("No active run. Start a run before logging.")
            return
        
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Could not log param {key}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to current run.
        
        Args:
            metrics: Dictionary of metric values
            step: Optional step number
        """
        if not self.current_run:
            logger.warning("No active run. Start a run before logging.")
            return
        
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, value, step=step)
            except Exception as e:
                logger.warning(f"Could not log metric {key}: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log artifact to current run.
        
        Args:
            local_path: Local file path
            artifact_path: Artifact directory in MLflow
        """
        if not self.current_run:
            logger.warning("No active run. Start a run before logging.")
            return
        
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"Could not log artifact: {e}")
    
    def log_dict(self, dictionary: Dict[str, Any], filename: str):
        """
        Log dictionary as JSON artifact.
        
        Args:
            dictionary: Dictionary to log
            filename: Filename for the artifact
        """
        if not self.current_run:
            logger.warning("No active run. Start a run before logging.")
            return
        
        try:
            mlflow.log_dict(dictionary, filename)
            logger.info(f"Logged dict as {filename}")
        except Exception as e:
            logger.error(f"Could not log dict: {e}")
    
    def log_text(self, text: str, filename: str):
        """
        Log text as artifact.
        
        Args:
            text: Text content
            filename: Filename for the artifact
        """
        if not self.current_run:
            logger.warning("No active run. Start a run before logging.")
            return
        
        try:
            mlflow.log_text(text, filename)
            logger.info(f"Logged text as {filename}")
        except Exception as e:
            logger.error(f"Could not log text: {e}")
    
    def set_tags(self, tags: Dict[str, str]):
        """
        Set tags on current run.
        
        Args:
            tags: Dictionary of tags
        """
        if not self.current_run:
            logger.warning("No active run. Start a run before setting tags.")
            return
        
        try:
            mlflow.set_tags(tags)
        except Exception as e:
            logger.error(f"Could not set tags: {e}")
    
    def get_run_info(self) -> Optional[Dict[str, Any]]:
        """Get information about current run."""
        if not self.current_run:
            return None
        
        return {
            "run_id": self.current_run.info.run_id,
            "run_name": self.current_run.data.tags.get("mlflow.runName"),
            "experiment_id": self.current_run.info.experiment_id,
            "status": self.current_run.info.status,
            "start_time": self.current_run.info.start_time,
            "artifact_uri": self.current_run.info.artifact_uri
        }


class RAGExperimentTracker:
    """
    Specialized tracker for RAG experiments.
    
    Tracks:
    - Retrieval configurations
    - Embedding models
    - Query performance
    - Retrieval quality metrics
    - System configurations
    """
    
    def __init__(
        self,
        mlflow_tracker: Optional[MLflowTracker] = None,
        experiment_name: str = "RAG_Experiments"
    ):
        """
        Initialize RAG experiment tracker.
        
        Args:
            mlflow_tracker: MLflow tracker instance
            experiment_name: Name of experiment
        """
        self.tracker = mlflow_tracker or MLflowTracker(
            config=ExperimentConfig(experiment_name=experiment_name)
        )
    
    def track_retrieval_experiment(
        self,
        run_name: str,
        retrieval_config: Dict[str, Any],
        embedding_model: str,
        metrics: Dict[str, float],
        queries: Optional[List[str]] = None,
        results: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Track a retrieval experiment.
        
        Args:
            run_name: Name for this run
            retrieval_config: Retrieval configuration parameters
            embedding_model: Name of embedding model
            metrics: Performance metrics
            queries: List of test queries
            results: Retrieval results
        """
        # Start run
        self.tracker.start_run(
            run_name=run_name,
            tags={
                "type": "retrieval",
                "embedding_model": embedding_model
            }
        )
        
        try:
            # Log parameters
            params = {
                "embedding_model": embedding_model,
                **retrieval_config
            }
            self.tracker.log_params(params)
            
            # Log metrics
            self.tracker.log_metrics(metrics)
            
            # Log queries if provided
            if queries:
                self.tracker.log_dict(
                    {"queries": queries},
                    "test_queries.json"
                )
            
            # Log results if provided
            if results:
                self.tracker.log_dict(
                    {"results": results},
                    "retrieval_results.json"
                )
            
            logger.info(f"Tracked retrieval experiment: {run_name}")
            
        finally:
            self.tracker.end_run()
    
    def track_rag_query(
        self,
        query: str,
        response: str,
        retrieved_docs: List[Dict[str, Any]],
        metrics: Dict[str, float],
        config: Dict[str, Any]
    ):
        """
        Track a single RAG query.
        
        Args:
            query: User query
            response: Generated response
            retrieved_docs: Retrieved documents
            metrics: Query metrics (latency, relevance, etc.)
            config: Query configuration
        """
        run_name = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.tracker.start_run(
            run_name=run_name,
            tags={"type": "query"}
        )
        
        try:
            # Log config
            self.tracker.log_params(config)
            
            # Log metrics
            self.tracker.log_metrics(metrics)
            
            # Log query and response
            self.tracker.log_dict(
                {
                    "query": query,
                    "response": response,
                    "retrieved_docs": retrieved_docs
                },
                "query_result.json"
            )
            
        finally:
            self.tracker.end_run()
    
    def track_benchmark(
        self,
        benchmark_name: str,
        benchmark_results: Dict[str, Any],
        config: Dict[str, Any]
    ):
        """
        Track benchmark results.
        
        Args:
            benchmark_name: Name of benchmark
            benchmark_results: Benchmark results dictionary
            config: Benchmark configuration
        """
        self.tracker.start_run(
            run_name=f"benchmark_{benchmark_name}",
            tags={"type": "benchmark"}
        )
        
        try:
            # Log config
            self.tracker.log_params(config)
            
            # Log summary metrics
            if "summary" in benchmark_results:
                self.tracker.log_metrics(benchmark_results["summary"])
            
            # Log full results
            self.tracker.log_dict(
                benchmark_results,
                "benchmark_results.json"
            )
            
            logger.info(f"Tracked benchmark: {benchmark_name}")
            
        finally:
            self.tracker.end_run()
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: Metrics to compare (all if None)
            
        Returns:
            Comparison results
        """
        comparison = {
            "runs": {},
            "metric_comparison": {}
        }
        
        for run_id in run_ids:
            try:
                run = self.tracker.client.get_run(run_id)
                
                comparison["runs"][run_id] = {
                    "name": run.data.tags.get("mlflow.runName", ""),
                    "params": run.data.params,
                    "metrics": run.data.metrics,
                    "start_time": run.info.start_time,
                    "status": run.info.status
                }
                
            except Exception as e:
                logger.warning(f"Could not get run {run_id}: {e}")
        
        # Compare metrics
        if metrics:
            for metric in metrics:
                comparison["metric_comparison"][metric] = {
                    run_id: run_data["metrics"].get(metric)
                    for run_id, run_data in comparison["runs"].items()
                }
        
        return comparison
    
    def get_best_run(
        self,
        metric_name: str,
        maximize: bool = True,
        filter_string: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get best run based on a metric.
        
        Args:
            metric_name: Name of metric to optimize
            maximize: Whether to maximize (True) or minimize (False)
            filter_string: Optional filter for runs
            
        Returns:
            Best run information
        """
        try:
            runs = self.tracker.client.search_runs(
                experiment_ids=[self.tracker.experiment_id],
                filter_string=filter_string,
                order_by=[f"metrics.{metric_name} {'DESC' if maximize else 'ASC'}"],
                max_results=1
            )
            
            if runs:
                best_run = runs[0]
                return {
                    "run_id": best_run.info.run_id,
                    "run_name": best_run.data.tags.get("mlflow.runName", ""),
                    "metric_value": best_run.data.metrics.get(metric_name),
                    "params": best_run.data.params,
                    "all_metrics": best_run.data.metrics
                }
            
        except Exception as e:
            logger.error(f"Error getting best run: {e}")
        
        return None


class MLflowCallback:
    """
    Callback for automatic MLflow logging during RAG operations.
    
    Can be integrated into RAG orchestrator for automatic tracking.
    """
    
    def __init__(self, tracker: RAGExperimentTracker):
        """
        Initialize callback.
        
        Args:
            tracker: RAG experiment tracker
        """
        self.tracker = tracker
        self.query_start_time = None
    
    def on_query_start(self, query: str, config: Dict[str, Any]):
        """Called when query starts."""
        import time
        self.query_start_time = time.time()
        self.current_query = query
        self.current_config = config
    
    def on_query_end(
        self,
        response: str,
        retrieved_docs: List[Dict[str, Any]],
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """Called when query ends."""
        import time
        
        if self.query_start_time is None:
            return
        
        # Calculate latency
        latency = time.time() - self.query_start_time
        
        # Prepare metrics
        metrics = {
            "latency_seconds": latency,
            "num_retrieved_docs": len(retrieved_docs)
        }
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Track query
        self.tracker.track_rag_query(
            query=self.current_query,
            response=response,
            retrieved_docs=retrieved_docs,
            metrics=metrics,
            config=self.current_config
        )
        
        # Reset
        self.query_start_time = None


def create_mlflow_ui_command(tracking_uri: Optional[str] = None) -> str:
    """
    Generate command to start MLflow UI.
    
    Args:
        tracking_uri: MLflow tracking URI
        
    Returns:
        Command string
    """
    cmd = "mlflow ui"
    
    if tracking_uri:
        cmd += f" --backend-store-uri {tracking_uri}"
    
    return cmd


def log_rag_system_info(
    tracker: MLflowTracker,
    system_config: Dict[str, Any]
):
    """
    Log RAG system configuration.
    
    Args:
        tracker: MLflow tracker
        system_config: System configuration dictionary
    """
    if tracker.current_run:
        tracker.log_params(system_config)
        tracker.log_dict(system_config, "system_config.json")


# Convenience decorators
def mlflow_track_function(
    tracker: MLflowTracker,
    run_name: Optional[str] = None,
    log_params: bool = True,
    log_result: bool = True
):
    """
    Decorator to automatically track function with MLflow.
    
    Args:
        tracker: MLflow tracker
        run_name: Run name (uses function name if None)
        log_params: Whether to log function parameters
        log_result: Whether to log function result
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            import time
            import inspect
            
            # Get function name
            func_name = run_name or func.__name__
            
            # Start run
            tracker.start_run(run_name=func_name)
            
            try:
                # Log parameters if requested
                if log_params:
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    
                    params = {
                        k: str(v) for k, v in bound_args.arguments.items()
                        if not k.startswith('_')
                    }
                    tracker.log_params(params)
                
                # Execute function
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                # Log metrics
                tracker.log_metrics({
                    "execution_time": end_time - start_time
                })
                
                # Log result if requested
                if log_result and result is not None:
                    tracker.log_dict(
                        {"result": str(result)},
                        "function_result.json"
                    )
                
                tracker.end_run(status="FINISHED")
                return result
                
            except Exception as e:
                tracker.end_run(status="FAILED")
                raise
        
        return wrapper
    return decorator
