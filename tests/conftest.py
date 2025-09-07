"""
Shared pytest fixtures for BentoML service testing
"""

import pytest
import bentoml
from pathlib import Path
import tempfile
import shutil
from typing import Generator


@pytest.fixture(scope="session")
def test_assets_dir() -> Path:
    """Path to test assets directory"""
    return Path(__file__).parent.parent / "test-assets"


@pytest.fixture(scope="session") 
def temp_models_dir() -> Generator[Path, None, None]:
    """Temporary directory for test models"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_image_path(test_assets_dir: Path) -> Path:
    """Path to sample test image"""
    return test_assets_dir / "test-office.jpg"


@pytest.fixture(scope="session") 
def sample_audio_path(test_assets_dir: Path) -> Path:
    """Path to sample test audio file"""
    return test_assets_dir / "test-english.mp3"


@pytest.fixture
def mock_model_response():
    """Mock model response for testing"""
    return {
        "status": "success",
        "message": "Mock response",
        "data": {"test": True}
    }


@pytest.fixture
def test_request_data():
    """Standard test request data"""
    return {"request": {"name": "test"}}


@pytest.fixture
def bentoml_client():
    """BentoML sync HTTP client for testing services"""
    # This will be configured per test to connect to specific services
    return None