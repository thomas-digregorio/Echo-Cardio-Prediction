import os
import pandas as pd
import torch
import numpy as np
from nvidia.dali import pipeline_def, fn, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


class EchoNetDALILoader:
    """
    NVIDIA DALI Loader for EchoNet-Dynamic.
    Optimized for RTX 5080 with Hardware Decoding (NVDEC).
    
    Phase 2 Updates:
    - Filter by split (TRAIN/VAL/TEST)
    - 32 frames instead of 16
    """
    def __init__(self, 
                 root_dir, 
                 file_list_csv, 
                 batch_size=4, 
                 num_threads=8,  # Increased for faster data loading 
                 device_id=0, 
                 sequence_length=16,  # VideoMAE-base pretrained with 16 frames
                 resize_size=224, 
                 random_shuffle=True,
                 training=True,
                 split="TRAIN"):  # New: filter by split
        
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.resize_size = resize_size
        self.device_id = device_id
        
        # Load File List
        self.df = pd.read_csv(file_list_csv)
        
        # Filter by split if specified
        if split is not None and 'Split' in self.df.columns:
            original_len = len(self.df)
            self.df = self.df[self.df['Split'] == split]
            print(f"Filtering by Split='{split}': {original_len} -> {len(self.df)} samples")
        
        # Filter: Ensure files exist
        self.video_files = []
        self.labels = []
        
        print(f"Initializing DALI Loader with {len(self.df)} potential files...")
        
        valid_count = 0
        for idx, row in self.df.iterrows():
            fname = row['FileName']
            if not fname.endswith('.avi'):
                fname += '.avi'
                
            full_path = os.path.join(self.root_dir, "Videos", fname)
            ef_label = row['EF']
            
            if os.path.exists(full_path):
                self.video_files.append(full_path)
                self.labels.append(np.array([ef_label], dtype=np.float32))
                valid_count += 1
            
        print(f"Found {valid_count} valid video files.")
        
        self.pipe = self._create_pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            video_files=self.video_files,
            labels=self.labels,
            sequence_length=sequence_length,
            random_shuffle=random_shuffle,
            training=training
        )
        
        self.pipe.build()
        
        self.dali_iter = DALIGenericIterator(
            self.pipe, 
            ['frames', 'label'], 
            last_batch_policy=LastBatchPolicy.PARTIAL if not training else LastBatchPolicy.DROP
        )

    @pipeline_def
    def _create_pipeline(self, video_files, labels, sequence_length, random_shuffle, training):
        """Defines the DALI Graph."""
        
        frames, label = fn.readers.video(
            device='gpu',
            filenames=video_files,
            labels=labels,
            sequence_length=sequence_length,
            shard_id=0,
            num_shards=1,
            random_shuffle=random_shuffle,
            initial_fill=1024,
            pad_last_batch=True,
            name='Reader',
            step=-1 if training else sequence_length, 
            dtype=types.UINT8
        )
        
        frames = fn.resize(
            frames, 
            resize_x=self.resize_size, 
            resize_y=self.resize_size,
            interp_type=types.INTERP_LINEAR
        )
        
        frames = fn.crop_mirror_normalize(
            frames,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            output_layout="FCHW",
            dtype=types.FLOAT
        )
        
        return frames, label

    def __iter__(self):
        return self.dali_iter.__iter__()

    def __len__(self):
        return len(self.video_files) // self.batch_size


def get_dataloader(data_dir, csv_path, batch_size=8, training=True, split="TRAIN"):
    """
    Get DALI dataloader with split filtering.
    
    Args:
        split: "TRAIN", "VAL", "TEST", or None (all data)
    """
    return EchoNetDALILoader(
        root_dir=data_dir,
        file_list_csv=csv_path,
        batch_size=batch_size,
        training=training,
        split=split
    )


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    possible_roots = [
        os.path.join(base_dir, "EchoNet-Dynamic", "EchoNet-Dynamic"),
        os.path.join(base_dir, "EchoNet-Dynamic"),
        os.path.join(base_dir, "data", "EchoNet-Dynamic")
    ]
    
    root = None
    for p in possible_roots:
        if os.path.exists(os.path.join(p, "FileList.csv")):
            root = p
            break
            
    if root is None:
        raise FileNotFoundError(f"Could not find EchoNet-Dynamic dataset")
    
    print(f"Detected Dataset Root: {root}")
    csv = os.path.join(root, "FileList.csv")
    
    # Test TRAIN split
    loader = get_dataloader(root, csv, batch_size=4, split="TRAIN")
    print("Fetching first batch...")
    
    for batch in loader:
        vid = batch[0]['frames']
        lbl = batch[0]['label']
        
        print(f"Batch Shape: {vid.shape}")  # Expect [4, 32, 3, 224, 224]
        print(f"Label Shape: {lbl.shape}")
        print(f"Device: {vid.device}")
        break
