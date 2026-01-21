from typing import List, Optional, Dict
from dataclasses import dataclass, field
import numpy as np
import time

@dataclass
class Episode:
    """Raw document chunk / experience."""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

@dataclass
class AKU:
    """Atomic Knowledge Unit - discrete factual proposition."""
    id: str
    proposition: str
    source_episode_ids: List[str]
    embedding: Optional[np.ndarray] = None
    structural_signature: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class GKU:
    """Global Knowledge Unit - conceptual cluster of AKUs."""
    id: str
    label: str
    aku_ids: List[str]
    centroid: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)
