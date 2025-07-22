from cil.framework import BlockDataContainer

class EnhancedBlockDataContainer(BlockDataContainer):
    """Enhanced BlockDataContainer with a max method for convenience."""

    def max(self) -> float:
        """Return the maximum value across all containers."""
        return max(d.max() for d in self.containers)