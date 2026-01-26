"""
Configuration management for the Credithos EKM system.
Centralizes all configuration parameters and provides validation.
"""
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import os
import json
from pathlib import Path
import yaml


@dataclass
class EKMConfig:
    """Configuration for the EKM system."""
    
    # Core dimensions
    embedding_dim: int = 768
    projection_dim: int = 64
    k_sparse: int = 10
    
    # Mathematical parameters
    alpha: float = 0.5  # Weight for semantic component
    beta: float = 0.3   # Weight for temporal component
    gamma: float = 0.1  # Weight for higher-order component
    tau: float = 86400  # Time decay constant (1 day in seconds)
    
    # Performance parameters
    mesh_threshold: int = 1000  # Threshold to switch to mesh mode
    candidate_size: int = 100   # Number of candidates for retrieval
    attention_temperature: float = 1.0
    
    # Service endpoints
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    
    # Advanced tensor settings
    enable_higher_order_terms: bool = True
    tensor_regularization: float = 0.01
    
    # Performance optimization
    use_scalable_index: bool = True
    batch_size: int = 100
    cache_enabled: bool = True
    cache_size: int = 1000
    
    # Validation parameters
    similarity_threshold: float = 0.1
    min_cluster_size: int = 2
    max_cluster_size: int = 50
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        
        if self.projection_dim <= 0:
            raise ValueError(f"projection_dim must be positive, got {self.projection_dim}")
        
        if self.k_sparse <= 0:
            raise ValueError(f"k_sparse must be positive, got {self.k_sparse}")
        
        if not (0 <= self.alpha <= 1):
            raise ValueError(f"alpha must be between 0 and 1, got {self.alpha}")
        
        if not (0 <= self.beta <= 1):
            raise ValueError(f"beta must be between 0 and 1, got {self.beta}")
        
        if not (0 <= self.gamma <= 1):
            raise ValueError(f"gamma must be between 0 and 1, got {self.gamma}")
        
        if self.tau <= 0:
            raise ValueError(f"tau must be positive, got {self.tau}")
        
        if self.mesh_threshold <= 0:
            raise ValueError(f"mesh_threshold must be positive, got {self.mesh_threshold}")
        
        if self.candidate_size <= 0:
            raise ValueError(f"candidate_size must be positive, got {self.candidate_size}")
        
        if self.attention_temperature <= 0:
            raise ValueError(f"attention_temperature must be positive, got {self.attention_temperature}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EKMConfig':
        """Create config from dictionary."""
        # Filter out keys that don't match the dataclass fields
        field_names = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'EKMConfig':
        """Load config from YAML or JSON file."""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def to_file(self, config_path: str, format: str = 'yaml'):
        """Save config to file."""
        path = Path(config_path)
        config_dict = self.to_dict()
        
        # Remove sensitive information from config dict
        sensitive_keys = ['qdrant_api_key', 'deepseek_api_key']
        for key in sensitive_keys:
            if key in config_dict:
                del config_dict[key]
        
        with open(path, 'w') as f:
            if format.lower() == 'yaml':
                yaml.dump(config_dict, f, default_flow_style=False)
            elif format.lower() == 'json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._config: Optional[EKMConfig] = None
    
    def load_config(self) -> EKMConfig:
        """Load configuration from file or environment."""
        if self._config is not None:
            return self._config
        
        # Try to load from file first
        if self.config_path:
            self._config = EKMConfig.from_file(self.config_path)
        else:
            # Load from environment variables or use defaults
            self._config = self._load_from_env()
        
        # Override with environment variables
        self._config = self._override_with_env(self._config)
        
        return self._config
    
    def _load_from_env(self) -> EKMConfig:
        """Load configuration from environment variables."""
        config_dict = {}
        
        # Map environment variables to config parameters
        env_mappings = {
            'EMBEDDING_DIM': 'embedding_dim',
            'PROJECTION_DIM': 'projection_dim',
            'K_SPARSE': 'k_sparse',
            'ALPHA': 'alpha',
            'BETA': 'beta',
            'GAMMA': 'gamma',
            'TAU': 'tau',
            'MESH_THRESHOLD': 'mesh_threshold',
            'CANDIDATE_SIZE': 'candidate_size',
            'ATTENTION_TEMPERATURE': 'attention_temperature',
            'QDRANT_URL': 'qdrant_url',
            'QDRANT_API_KEY': 'qdrant_api_key',
            'DEEPSEEK_API_KEY': 'deepseek_api_key',
            'ENABLE_HIGHER_ORDER_TERMS': 'enable_higher_order_terms',
            'TENSOR_REGULARIZATION': 'tensor_regularization',
            'USE_SCALABLE_INDEX': 'use_scalable_index',
            'BATCH_SIZE': 'batch_size',
            'CACHE_ENABLED': 'cache_enabled',
            'CACHE_SIZE': 'cache_size',
            'SIMILARITY_THRESHOLD': 'similarity_threshold',
            'MIN_CLUSTER_SIZE': 'min_cluster_size',
            'MAX_CLUSTER_SIZE': 'max_cluster_size'
        }
        
        for env_var, config_param in env_mappings.items():
            env_val = os.getenv(env_var)
            if env_val is not None:
                # Convert string values to appropriate types
                field_type = EKMConfig.__dataclass_fields__[config_param].type
                converted_val = self._convert_type(env_val, field_type)
                config_dict[config_param] = converted_val
        
        return EKMConfig(**config_dict)
    
    def _convert_type(self, value: str, target_type: type) -> Any:
        """Convert string value to target type."""
        if target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif target_type == str:
            return value
        elif target_type == type(None):
            return value
        else:
            # Handle Optional types
            if hasattr(target_type, '__origin__') and target_type.__origin__ is type(None):
                return value
            return value  # Default to string
    
    def _override_with_env(self, config: EKMConfig) -> EKMConfig:
        """Override config values with environment variables."""
        # Check for environment variables that should override config
        qdrant_url = os.getenv('QDRANT_URL')
        if qdrant_url:
            config.qdrant_url = qdrant_url
        
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        if qdrant_api_key:
            config.qdrant_api_key = qdrant_api_key
        
        deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        if deepseek_api_key:
            config.deepseek_api_key = deepseek_api_key
        
        return config


# Global config manager instance
_config_manager = ConfigManager()


def get_config(config_path: Optional[str] = None) -> EKMConfig:
    """Get the global configuration."""
    if config_path:
        return ConfigManager(config_path).load_config()
    else:
        return _config_manager.load_config()


def load_config(config_path: Optional[str] = None) -> EKMConfig:
    """Load configuration from file or environment."""
    return get_config(config_path)