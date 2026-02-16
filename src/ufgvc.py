import os
import requests
import pandas as pd
from PIL import Image
import io
from torch.utils.data import Dataset
import torch
from typing import Optional, Callable, Tuple, Any, Dict, Union
from pathlib import Path
import hashlib

class UFGVCDataset(Dataset):
    """
    UFGVC Dataset - A PyTorch-like dataset for various agricultural classification tasks
    
    Supports multiple datasets from the UFGVC collection with automatic download.
    
    Args:
        dataset_name (str): Name of the dataset to use
        root (str): Root directory where dataset will be stored
        split (str): Dataset split - 'train', 'test', or 'val'
        transform (callable, optional): Transform to be applied on images
        target_transform (callable, optional): Transform to be applied on labels
        download (bool): If True, downloads the dataset if not found locally
    """
    
    # Available datasets configuration
    DATASETS = {
        'cotton80': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/cotton80_dataset.parquet?download=true',
            'filename': 'cotton80_dataset.parquet',
            'description': 'Cotton classification dataset with 80 classes'
        },
        'soybean': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/soybean_dataset.parquet?download=true',
            'filename': 'soybean_dataset.parquet',
            'description': 'Soybean classification dataset'
        },
        'soy_ageing_r1': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/soy_ageing_R1_dataset.parquet?download=true',
            'filename': 'soy_ageing_R1_dataset.parquet',
            'description': 'Soybean ageing dataset - Round 1'
        },
        'soy_ageing_r3': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/soy_ageing_R3_dataset.parquet?download=true',
            'filename': 'soy_ageing_R3_dataset.parquet',
            'description': 'Soybean ageing dataset - Round 3'
        },
        'soy_ageing_r4': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/soy_ageing_R4_dataset.parquet?download=true',
            'filename': 'soy_ageing_R4_dataset.parquet',
            'description': 'Soybean ageing dataset - Round 4'
        },
        'soy_ageing_r5': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/soy_ageing_R5_dataset.parquet?download=true',
            'filename': 'soy_ageing_R5_dataset.parquet',
            'description': 'Soybean ageing dataset - Round 5'
        },
        'soy_ageing_r6': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/soy_ageing_R6_dataset.parquet?download=true',
            'filename': 'soy_ageing_R6_dataset.parquet',
            'description': 'Soybean ageing dataset - Round 6'
        },
        'cub_200_2011': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/cub_dataset.parquet?download=true',
            'filename': 'cub_dataset.parquet',
            'description': 'CUB-200-2011 dataset for fine-grained bird classification'
        },
        'soygene': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/soygene_dataset.parquet?download=true',
            'filename': 'soygene_dataset.parquet',
            'description': 'Soygene dataset for soybean classification'
        },
        'soyglobal': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/soyglobal_dataset.parquet?download=true',
            'filename': 'soyglobal_dataset.parquet',
            'description': 'Soyglobal dataset for global soybean classification'
        },
        'stanford_cars': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/stanford_cars_dataset.parquet?download=true',
            'filename': 'stanford_cars_dataset.parquet',
            'description': 'Stanford Cars dataset for fine-grained car classification'
        },
        'nabirds': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/nabirds_dataset.parquet?download=true',
            'filename': 'nabirds_dataset.parquet',
            'description': 'NABirds dataset for fine-grained bird classification'
        },
        'fgvc_aircraft': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/fgvcaircraft_variant.parquet?download=true',
            'filename': 'fgvcaircraft_variant.parquet',
            'description': 'FGVC Aircraft dataset for fine-grained aircraft classification'
        },
        'food101': {
            'url': '',
            'filename': 'food101_subset.parquet',
            'description': 'Food-101 for food classification'
        },
        'flowers102': {
            'url': '',
            'filename': 'flowers102_subset.parquet',
            'description': 'Oxford Flowers 102 for flower classification'
        },
        'oxford_pets': {
            'url': '',
            'filename': 'oxford_pets.parquet',
            'description': 'Oxford-IIIT Pets for pet classification'
        }
    }
    
    def __init__(
        self,
        dataset_name: str = "cotton80",
        root: str = "./data",
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        return_index: bool = False,
    ):
        if dataset_name not in self.DATASETS:
            available = list(self.DATASETS.keys())
            raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {available}")
        
        self.dataset_name = dataset_name
        self.dataset_config = self.DATASETS[dataset_name]
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.return_index = return_index
        
        # Dataset file info
        self.url = self.dataset_config['url']
        self.filename = self.dataset_config['filename']
        self.filepath = self.root / self.filename
        
        # Ensure root directory exists
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Download if needed
        if download and not self.filepath.exists():
            self._download()
        
        # Load and filter data
        self._load_data()
        
    def _download(self):
        """Download the dataset file"""
        print(f"Downloading {self.dataset_name} dataset...")
        print(f"Description: {self.dataset_config['description']}")
        print(f"Saving to: {self.filepath}")
        
        try:
            response = requests.get(self.url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(self.filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\rProgress: {progress:.1f}%", end="")
            
            print(f"\nDownload completed: {self.filepath}")
            
        except Exception as e:
            if self.filepath.exists():
                self.filepath.unlink()  # Remove incomplete file
            raise RuntimeError(f"Failed to download {self.dataset_name}: {e}")
    
    def _load_data(self):
        """Load and filter the parquet data"""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.filepath}")
        
        try:
            # Load parquet file
            df = pd.read_parquet(self.filepath)
            
            # Validate columns
            expected_cols = {'image', 'label', 'class_name', 'split'}
            if not expected_cols.issubset(df.columns):
                missing_cols = expected_cols - set(df.columns)
                raise ValueError(f"Missing columns: {missing_cols}")
            
            # Filter by split
            self.data = df[df['split'] == self.split].reset_index(drop=True)
            
            if len(self.data) == 0:
                available_splits = df['split'].unique().tolist()
                raise ValueError(f"No data found for split '{self.split}'. Available splits: {available_splits}")
            
            # IMPORTANT: build a single, stable label mapping from the FULL dataset
            # so train/val/test splits share the same class_to_idx.
            unique_classes = df['class_name'].unique()
            try:
                # Try to sort numerically if all class names are numeric
                self.classes = sorted(unique_classes, key=int)
            except ValueError:
                # Fall back to string sorting if not all numeric
                self.classes = sorted(unique_classes)
            
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            
            print(f"Dataset: {self.dataset_name}")
            print(f"Split: {self.split}")
            print(f"Samples: {len(self.data)}")
            print(f"Classes: {len(self.classes)}")
            # Avoid printing full class lists/mappings (can be very large)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {self.dataset_name}: {e}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Union[Tuple[Any, Any], Tuple[Any, Any, int]]:
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        # Get row data
        row = self.data.iloc[idx]
        
        # Load image from bytes
        image_bytes = row['image']
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image at index {idx}: {e}")
        
        # Get label - use class_name to ensure consistency with self.classes
        class_name = row['class_name']
        label = self.class_to_idx[class_name]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        if self.return_index:
            return image, label, idx
        return image, label

    def get_all_labels(self) -> torch.Tensor:
        """Return all labels in dataset order as a 1D LongTensor.

        This does NOT decode images and is safe to call for alpha statistics.
        """
        class_names = self.data["class_name"].tolist()
        labels = [self.class_to_idx[name] for name in class_names]
        return torch.tensor(labels, dtype=torch.long)
    
    def get_class_name(self, idx: int) -> str:
        """Get class name for a given index"""
        return self.data.iloc[idx]['class_name']
    
    def get_dataset_info(self) -> dict:
        """Get comprehensive information about the dataset"""
        # Get split distribution
        full_df = pd.read_parquet(self.filepath)
        split_counts = full_df['split'].value_counts().to_dict()
        total_classes = len(full_df['class_name'].unique())
        
        return {
            'dataset_name': self.dataset_name,
            'description': self.dataset_config['description'],
            'current_split': self.split,
            'current_samples': len(self.data),
            'current_classes': len(self.classes),
            'total_samples': len(full_df),
            'total_classes': total_classes,
            'split_distribution': split_counts,
            'classes': self.classes,
            'filepath': str(self.filepath)
        }
    
    def get_sample_info(self, idx: int) -> dict:
        """Get detailed information about a specific sample"""
        row = self.data.iloc[idx]
        return {
            'dataset': self.dataset_name,
            'index': idx,
            'label': int(row['label']),
            'class_name': row['class_name'],
            'split': row['split']
        }
    
    @classmethod
    def list_available_datasets(cls) -> Dict[str, str]:
        """List all available datasets with descriptions"""
        return {name: config['description'] for name, config in cls.DATASETS.items()}
    
    @classmethod
    def get_dataset_splits(cls, dataset_name: str, root: str = "./data") -> list:
        """Get available splits for a specific dataset"""
        if dataset_name not in cls.DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        config = cls.DATASETS[dataset_name]
        filepath = Path(root) / config['filename']
        
        if not filepath.exists():
            print(f"Dataset file not found. Available splits unknown until downloaded.")
            return []
        
        try:
            df = pd.read_parquet(filepath)
            return df['split'].unique().tolist()
        except Exception as e:
            print(f"Error reading dataset: {e}")
            return []


# Enhanced utility functions
def create_multi_dataloaders(
    dataset_names: list,
    root: str = "./data",
    batch_size: int = 32,
    num_workers: int = 4,
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None,
    download: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Create DataLoaders for multiple datasets and splits
    
    Args:
        dataset_names: List of dataset names to load
        
    Returns:
        dict: Nested dictionary {dataset_name: {split: DataLoader}}
    """
    from torch.utils.data import DataLoader
    
    all_dataloaders = {}
    
    for dataset_name in dataset_names:
        print(f"\n=== Loading {dataset_name} ===")
        dataloaders = {}
        
        # Try to get available splits
        available_splits = UFGVCDataset.get_dataset_splits(dataset_name, root)
        if not available_splits:
            available_splits = ['train', 'val', 'test']  # Default attempt
        
        for split in available_splits:
            try:
                transform = transform_train if split == 'train' else transform_val
                dataset = UFGVCDataset(
                    dataset_name=dataset_name,
                    root=root,
                    split=split,
                    transform=transform,
                    download=download
                )
                
                shuffle = (split == 'train')
                dataloaders[split] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=True
                )
                
            except ValueError as e:
                print(f"Warning: Could not create dataset for {dataset_name}-{split}: {e}")
                continue
        
        if dataloaders:
            all_dataloaders[dataset_name] = dataloaders
    
    return all_dataloaders


def compare_datasets(dataset_names: list, root: str = "./data") -> pd.DataFrame:
    """Compare multiple datasets and return summary statistics"""
    comparison_data = []
    
    for dataset_name in dataset_names:
        try:
            # Create a temporary dataset to get info
            temp_dataset = UFGVCDataset(dataset_name=dataset_name, root=root, download=True)
            info = temp_dataset.get_dataset_info()
            
            comparison_data.append({
                'Dataset': dataset_name,
                'Description': info['description'],
                'Total Samples': info['total_samples'],
                'Total Classes': info['total_classes'],
                'Train Samples': info['split_distribution'].get('train', 0),
                'Val Samples': info['split_distribution'].get('val', 0),
                'Test Samples': info['split_distribution'].get('test', 0),
                'Available Splits': list(info['split_distribution'].keys())
            })
            
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            continue
    
    return pd.DataFrame(comparison_data)


# Example usage
if __name__ == "__main__":
    from torchvision import transforms
    
    # List available datasets
    print("Available datasets:")
    for name, desc in UFGVCDataset.list_available_datasets().items():
        print(f"  - {name}: {desc}")
    
    # Define transforms
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Example 1: Single dataset
    print("\n=== Single Dataset Example ===")
    dataset = UFGVCDataset(
        dataset_name="soybean",
        root="./data",
        split="train",
        transform=transform_train,
        download=True
    )
    
    print(dataset.get_dataset_info())
    
    # Example 2: Multiple datasets
    print("\n=== Multiple Datasets Example ===")
    dataset_names = ['cotton80', 'soybean', 'soy_ageing_r1']
    
    # Compare datasets
    comparison_df = compare_datasets(dataset_names)
    print("\nDataset Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Create DataLoaders for multiple datasets
    all_dataloaders = create_multi_dataloaders(
        dataset_names=dataset_names,
        root="./data",
        batch_size=16,
        transform_train=transform_train,
        transform_val=transform_val
    )
    
    print(f"\nCreated DataLoaders for: {list(all_dataloaders.keys())}")
    
    # Test a specific DataLoader
    if 'soybean' in all_dataloaders and 'train' in all_dataloaders['soybean']:
        loader = all_dataloaders['soybean']['train']
        print(f"Soybean train batches: {len(loader)}")
        
        # Get first batch
        for batch_images, batch_labels in loader:
            print(f"Batch shape: {batch_images.shape}")
            print(f"Labels shape: {batch_labels.shape}")
            break