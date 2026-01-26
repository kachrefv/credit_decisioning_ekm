import pytest
import asyncio
from typing import AsyncGenerator, Generator
from fastapi.testclient import TestClient
from ekm.api.app import app
from ekm.infra.auth import get_current_user

# Mock authentication to bypass security for testing (unless testing auth specifically)
async def mock_get_current_user():
    return {"sub": "test@example.com", "role": "admin"}

@pytest.fixture(scope="module")
def client() -> Generator:
    # Use TestClient for synchronous testing of FastAPI app
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="module")
def authenticated_client() -> Generator:
    # Override the auth dependency
    app.dependency_overrides[get_current_user] = mock_get_current_user
    with TestClient(app) as c:
        yield c
    # Clean up override
    app.dependency_overrides = {}

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
