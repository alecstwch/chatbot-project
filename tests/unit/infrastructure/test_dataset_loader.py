"""
Unit tests for dataset loader.

Tests the DatasetLoader class for downloading, caching, and validating
HuggingFace datasets following DDD and 12-Factor App principles.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datasets import DatasetDict, Dataset

from src.infrastructure.data.dataset_loader import DatasetLoader


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory for testing."""
    return tmp_path / "data"


@pytest.fixture
def dataset_loader(temp_data_dir):
    """Create DatasetLoader instance with temp directory."""
    return DatasetLoader(temp_data_dir)


@pytest.fixture
def mock_dataset():
    """Create mock HuggingFace dataset."""
    train_data = Dataset.from_dict({
        'text': ['Hello', 'How are you?', 'I am fine'],
        'label': [0, 1, 0]
    })
    test_data = Dataset.from_dict({
        'text': ['Test message'],
        'label': [1]
    })
    return DatasetDict({
        'train': train_data,
        'test': test_data
    })


class TestDatasetLoaderInitialization:
    """Test DatasetLoader initialization."""
    
    def test_init_creates_directories(self, temp_data_dir):
        """Test that initialization creates necessary directories."""
        loader = DatasetLoader(temp_data_dir)
        
        assert loader.data_dir == temp_data_dir
        assert loader.raw_dir == temp_data_dir / "raw"
        assert loader.raw_dir.exists()
    
    def test_init_with_existing_directory(self, temp_data_dir):
        """Test initialization with pre-existing directory."""
        temp_data_dir.mkdir(parents=True)
        (temp_data_dir / "raw").mkdir()
        
        loader = DatasetLoader(temp_data_dir)
        assert loader.raw_dir.exists()


class TestTherapyDatasetLoading:
    """Test loading therapy datasets."""
    
    @patch('src.infrastructure.data.dataset_loader.load_dataset')
    def test_load_therapy_dataset_downloads_when_not_cached(
        self, 
        mock_load, 
        dataset_loader, 
        mock_dataset
    ):
        """Test downloading therapy dataset when not cached."""
        mock_load.return_value = mock_dataset
        
        result = dataset_loader.load_therapy_dataset()
        
        mock_load.assert_called_once_with("Amod/mental_health_counseling_conversations")
        assert isinstance(result, DatasetDict)
        assert 'train' in result
    
    @patch('src.infrastructure.data.dataset_loader.load_dataset')
    def test_load_therapy_dataset_uses_cache(
        self, 
        mock_load, 
        dataset_loader, 
        mock_dataset
    ):
        """Test loading from cache when available."""
        # First call - download and save
        mock_load.return_value = mock_dataset
        dataset_loader.load_therapy_dataset()
        
        # Second call - should use cache
        mock_load.reset_mock()
        result = dataset_loader.load_therapy_dataset()
        
        # load_dataset should not be called again
        mock_load.assert_not_called()
        assert isinstance(result, DatasetDict)
    
    @patch('src.infrastructure.data.dataset_loader.load_dataset')
    def test_load_therapy_dataset_force_download(
        self, 
        mock_load, 
        dataset_loader, 
        mock_dataset
    ):
        """Test forcing re-download even when cached."""
        # First call
        mock_load.return_value = mock_dataset
        dataset_loader.load_therapy_dataset()
        
        # Force download
        mock_load.reset_mock()
        dataset_loader.load_therapy_dataset(force_download=True)
        
        mock_load.assert_called_once()
    
    @patch('src.infrastructure.data.dataset_loader.load_dataset')
    def test_load_therapy_dataset_custom_name(
        self, 
        mock_load, 
        dataset_loader, 
        mock_dataset
    ):
        """Test loading with custom dataset name."""
        mock_load.return_value = mock_dataset
        custom_name = "custom/therapy-dataset"
        
        dataset_loader.load_therapy_dataset(dataset_name=custom_name)
        
        mock_load.assert_called_once_with(custom_name)
    
    @patch('src.infrastructure.data.dataset_loader.load_dataset')
    def test_load_therapy_dataset_handles_errors(
        self, 
        mock_load, 
        dataset_loader
    ):
        """Test error handling when dataset loading fails."""
        mock_load.side_effect = Exception("Network error")
        
        with pytest.raises(ValueError, match="Failed to load therapy dataset"):
            dataset_loader.load_therapy_dataset()


class TestDialogDatasetLoading:
    """Test loading dialog datasets."""
    
    @patch('src.infrastructure.data.dataset_loader.load_dataset')
    def test_load_dialog_dataset_downloads_when_not_cached(
        self, 
        mock_load, 
        dataset_loader, 
        mock_dataset
    ):
        """Test downloading dialog dataset when not cached."""
        mock_load.return_value = mock_dataset
        
        result = dataset_loader.load_dialog_dataset()
        
        mock_load.assert_called_once_with("daily_dialog")
        assert isinstance(result, DatasetDict)
    
    @patch('src.infrastructure.data.dataset_loader.load_dataset')
    def test_load_dialog_dataset_uses_cache(
        self, 
        mock_load, 
        dataset_loader, 
        mock_dataset
    ):
        """Test loading from cache when available."""
        mock_load.return_value = mock_dataset
        dataset_loader.load_dialog_dataset()
        
        mock_load.reset_mock()
        result = dataset_loader.load_dialog_dataset()
        
        mock_load.assert_not_called()
        assert isinstance(result, DatasetDict)
    
    @patch('src.infrastructure.data.dataset_loader.load_dataset')
    def test_load_dialog_dataset_custom_name(
        self, 
        mock_load, 
        dataset_loader, 
        mock_dataset
    ):
        """Test loading with custom dataset name."""
        mock_load.return_value = mock_dataset
        custom_name = "custom/dialog-dataset"
        
        dataset_loader.load_dialog_dataset(dataset_name=custom_name)
        
        mock_load.assert_called_once_with(custom_name)
    
    @patch('src.infrastructure.data.dataset_loader.load_dataset')
    def test_load_dialog_dataset_handles_errors(
        self, 
        mock_load, 
        dataset_loader
    ):
        """Test error handling for dialog dataset."""
        mock_load.side_effect = Exception("Connection timeout")
        
        with pytest.raises(ValueError, match="Failed to load dialog dataset"):
            dataset_loader.load_dialog_dataset()


class TestLoadAllDatasets:
    """Test loading all datasets at once."""
    
    @patch('src.infrastructure.data.dataset_loader.load_dataset')
    def test_load_all_datasets(
        self, 
        mock_load, 
        dataset_loader, 
        mock_dataset
    ):
        """Test loading both datasets."""
        mock_load.return_value = mock_dataset
        
        therapy_data, dialog_data = dataset_loader.load_all_datasets()
        
        assert isinstance(therapy_data, DatasetDict)
        assert isinstance(dialog_data, DatasetDict)
        assert mock_load.call_count == 2
    
    @patch('src.infrastructure.data.dataset_loader.load_dataset')
    def test_load_all_datasets_with_force_download(
        self, 
        mock_load, 
        dataset_loader, 
        mock_dataset
    ):
        """Test force downloading both datasets."""
        mock_load.return_value = mock_dataset
        
        # First load
        dataset_loader.load_all_datasets()
        
        # Force reload
        mock_load.reset_mock()
        dataset_loader.load_all_datasets(force_download=True)
        
        assert mock_load.call_count == 2


class TestDatasetInfo:
    """Test dataset information extraction."""
    
    def test_get_dataset_info(self, dataset_loader, mock_dataset):
        """Test extracting dataset split information."""
        info = dataset_loader.get_dataset_info(mock_dataset)
        
        assert 'train' in info
        assert 'test' in info
        assert info['train'] == 3  # 3 samples in train split
        assert info['test'] == 1   # 1 sample in test split
    
    def test_get_dataset_info_empty_dataset(self, dataset_loader):
        """Test with empty dataset."""
        empty_dataset = DatasetDict()
        info = dataset_loader.get_dataset_info(empty_dataset)
        
        assert info == {}


class TestDatasetValidation:
    """Test dataset validation."""
    
    def test_validate_dataset_with_required_splits(
        self, 
        dataset_loader, 
        mock_dataset
    ):
        """Test validation passes with required splits."""
        assert dataset_loader.validate_dataset(mock_dataset, ['train'])
        assert dataset_loader.validate_dataset(mock_dataset, ['train', 'test'])
    
    def test_validate_dataset_missing_split(
        self, 
        dataset_loader, 
        mock_dataset
    ):
        """Test validation fails with missing split."""
        result = dataset_loader.validate_dataset(
            mock_dataset, 
            ['train', 'validation']
        )
        assert result is False
    
    def test_validate_dataset_default_splits(
        self, 
        dataset_loader, 
        mock_dataset
    ):
        """Test validation with default splits (train only)."""
        assert dataset_loader.validate_dataset(mock_dataset)
    
    def test_validate_dataset_empty(self, dataset_loader):
        """Test validation with empty dataset."""
        empty_dataset = DatasetDict()
        assert dataset_loader.validate_dataset(empty_dataset) is False


class TestDatasetPersistence:
    """Test dataset saving and loading from disk."""
    
    @patch('src.infrastructure.data.dataset_loader.load_dataset')
    def test_dataset_saved_to_correct_location(
        self, 
        mock_load, 
        dataset_loader, 
        mock_dataset
    ):
        """Test dataset is saved to correct directory."""
        mock_load.return_value = mock_dataset
        
        dataset_loader.load_therapy_dataset()
        
        therapy_path = dataset_loader.raw_dir / "therapy"
        assert therapy_path.exists()
    
    @patch('src.infrastructure.data.dataset_loader.load_dataset')
    def test_both_datasets_saved_separately(
        self, 
        mock_load, 
        dataset_loader, 
        mock_dataset
    ):
        """Test both datasets saved in separate directories."""
        mock_load.return_value = mock_dataset
        
        dataset_loader.load_all_datasets()
        
        therapy_path = dataset_loader.raw_dir / "therapy"
        dialog_path = dataset_loader.raw_dir / "dialogs"
        
        assert therapy_path.exists()
        assert dialog_path.exists()
