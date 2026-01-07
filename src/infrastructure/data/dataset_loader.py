"""
Dataset loader for downloading and managing HuggingFace datasets.

This module provides functionality to download therapy and dialog datasets
from HuggingFace and save them locally following 12-Factor App principles.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from datasets import load_dataset, DatasetDict

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads and manages datasets from HuggingFace Hub.
    
    Follows 12-Factor App principles by:
    - Treating data as attached resources
    - Storing datasets in configurable locations
    - Proper error handling and logging
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Base directory for storing datasets
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
    def load_therapy_dataset(
        self, 
        dataset_name: str = "Amod/mental_health_counseling_conversations",
        force_download: bool = False
    ) -> DatasetDict:
        """
        Load mental health counseling dataset.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            force_download: If True, re-download even if cached
            
        Returns:
            DatasetDict with train/validation/test splits
            
        Raises:
            ValueError: If dataset cannot be loaded
        """
        save_path = self.raw_dir / "therapy"
        
        try:
            if save_path.exists() and not force_download:
                logger.info(f"Loading cached therapy dataset from {save_path}")
                dataset = DatasetDict.load_from_disk(str(save_path))
            else:
                logger.info(f"Downloading therapy dataset: {dataset_name}")
                dataset = load_dataset(dataset_name)
                
                # Save to disk
                logger.info(f"Saving therapy dataset to {save_path}")
                dataset.save_to_disk(str(save_path))
                
            logger.info(f"Therapy dataset loaded: {len(dataset)} splits")
            return dataset
            
        except Exception as e:
            error_msg = f"Failed to load therapy dataset: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    def load_dialog_dataset(
        self,
        dataset_name: str = "daily_dialog",
        force_download: bool = False
    ) -> DatasetDict:
        """
        Load daily dialog conversational dataset.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            force_download: If True, re-download even if cached
            
        Returns:
            DatasetDict with train/validation/test splits
            
        Raises:
            ValueError: If dataset cannot be loaded
        """
        save_path = self.raw_dir / "dialogs"
        
        try:
            if save_path.exists() and not force_download:
                logger.info(f"Loading cached dialog dataset from {save_path}")
                dataset = DatasetDict.load_from_disk(str(save_path))
            else:
                logger.info(f"Downloading dialog dataset: {dataset_name}")
                dataset = load_dataset(dataset_name)
                
                # Save to disk
                logger.info(f"Saving dialog dataset to {save_path}")
                dataset.save_to_disk(str(save_path))
                
            logger.info(f"Dialog dataset loaded: {len(dataset)} splits")
            return dataset
            
        except Exception as e:
            error_msg = f"Failed to load dialog dataset: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    def load_all_datasets(
        self,
        force_download: bool = False
    ) -> Tuple[DatasetDict, DatasetDict]:
        """
        Load both therapy and dialog datasets.
        
        Args:
            force_download: If True, re-download even if cached
            
        Returns:
            Tuple of (therapy_dataset, dialog_dataset)
        """
        logger.info("Loading all datasets...")
        therapy_data = self.load_therapy_dataset(force_download=force_download)
        dialog_data = self.load_dialog_dataset(force_download=force_download)
        logger.info("All datasets loaded successfully")
        return therapy_data, dialog_data
    
    def get_dataset_info(self, dataset: DatasetDict) -> Dict[str, int]:
        """
        Get basic information about a dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary with split names and sizes
        """
        info = {}
        for split_name, split_data in dataset.items():
            info[split_name] = len(split_data)
        return info
    
    def validate_dataset(self, dataset: DatasetDict, required_splits: Optional[list] = None) -> bool:
        """
        Validate that dataset has required structure.
        
        Args:
            dataset: Dataset to validate
            required_splits: List of required split names (default: ['train'])
            
        Returns:
            True if valid, False otherwise
        """
        if required_splits is None:
            required_splits = ['train']
            
        for split in required_splits:
            if split not in dataset:
                logger.warning(f"Missing required split: {split}")
                return False
                
        return True
