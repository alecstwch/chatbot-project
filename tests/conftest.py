"""
Pytest configuration and fixtures

Shared fixtures and configuration for all tests.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def test_data_dir():
    """Get test data directory"""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_texts():
    """Sample texts for testing preprocessing"""
    return [
        "Hello! How are you doing today?",
        "I'm feeling very anxious about my health.",
        "Check this link: https://example.com for more info.",
        "This is a <b>test</b> with HTML tags.",
        "Contact me at user@example.com",
    ]


@pytest.fixture
def sample_conversation():
    """Sample conversation for testing"""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "I'm feeling anxious"},
        {"role": "assistant", "content": "I understand. Can you tell me more about what's making you feel anxious?"},
    ]


@pytest.fixture(scope="function")
def temp_dir(tmp_path):
    """Temporary directory for test files"""
    return tmp_path
