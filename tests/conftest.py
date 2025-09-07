"""
Shared pytest fixtures for BentoML service testing
"""

import pytest
import bentoml
import os
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


@pytest.fixture(scope="session")
def bentoml_host():
    """BentoML server host from environment"""
    return os.getenv("BENTOML_HOST", "127.0.0.1")


@pytest.fixture(scope="session")
def bentoml_port():
    """BentoML server port from environment"""
    return os.getenv("BENTOML_PORT", "3000")


@pytest.fixture(scope="session")
def bentoml_protocol():
    """BentoML server protocol from environment"""
    return os.getenv("BENTOML_PROTOCOL", "http")


@pytest.fixture(scope="session")
def bentoml_base_url(bentoml_protocol: str, bentoml_host: str, bentoml_port: str):
    """Base URL for BentoML service"""
    return f"{bentoml_protocol}://{bentoml_host}:{bentoml_port}"


def get_service_url(service_name: str, default_port: str = "3000") -> str:
    """Get service URL from environment variables"""
    protocol = os.getenv("BENTOML_PROTOCOL", "http")
    host = os.getenv("BENTOML_HOST", "127.0.0.1")
    port = os.getenv(f"{service_name.upper()}_SERVICE_PORT", default_port)
    return f"{protocol}://{host}:{port}"


@pytest.fixture
def bentoml_client():
    """BentoML sync HTTP client for testing services"""
    # This will be configured per test to connect to specific services
    return None