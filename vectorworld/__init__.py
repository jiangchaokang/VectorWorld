"""VectorWorld: Efficient Streaming World Model via Diffusion Flow on Vector Graphs.

Backward-compatible import aliases for checkpoint loading.
"""

# Aliases for checkpoint loading compatibility
# Lightning looks up the original class path when loading checkpoints.
# These aliases ensure old checkpoint paths resolve correctly.

from vectorworld.models.vae import VectorWorldVAE as ScenarioDreamerAutoEncoder
from vectorworld.models.ldm import VectorWorldLDM as ScenarioDreamerLDM
from vectorworld.models.delta_sim import DeltaSim as CtRLSim