"""
Shared fixtures and configuration for flash_attn_legacy tests.
"""

import pytest
import torch


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (use -m slow to run)")


@pytest.fixture(scope="session")
def device_info():
    """GPU info, available for the entire test session."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "sm": props.major * 10 + props.minor,
        "mem_gb": round(props.total_memory / (1024**3), 1),
    }


@pytest.fixture(autouse=True)
def cuda_sync():
    """Sync CUDA after each test to catch async errors."""
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
