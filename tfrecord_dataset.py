# tfrecord_dataset.py
# Dataset implementation for reading ImageNet from TFRecords on Kaggle (e.g., imagenet-1k-tfrecords-ilsvrc2012-part-*)

import os
from pathlib import Path
import torch
from torch.utils.data import IterableDataset
import tensorflow as tf
from PIL import Image
import io

# --- TFRecord Parsing Logic (Based on Kaggle notebook patterns) ---

def _parse_tfrecord_fn(example, subset='train'):
    """Parses a single TFRecord example for ImageNet data.
       Adjust feature keys if your TFRecords use different names.
    """
    # Define the features available in your TFRecords
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/class/label': tf.io.FixedLenFeature([], tf.int64),
        # 'image/class/text': tf.io.FixedLenFeature([], tf.string), # Optional
        # Add other features if present
    }

    # Parse the example
    parsed_features = tf.io.parse_single_example(example, feature_description)

    # --- Image Decoding ---
    image_encoded = parsed_features['image/encoded']
    # Decode JPEG image
    image = tf.io.decode_jpeg(image_encoded, channels=3) # Assumes JPEG and RGB
    # Convert to float32 and potentially normalize if needed by your model/augmentations
    # The original script's transforms handle normalization (mean/std), so we just convert to float
    image = tf.cast(image, tf.float32) # Range [0, 255] as float

    # --- Label Handling ---
    label = parsed_features['image/class/label']
    # Important: TFRecord labels are often 1-based (1-1000 for ImageNet).
    # PyTorch models typically expect 0-based labels (0-999).
    # Adjust if your labels are already 0-based.
    label = label - 1 # Convert 1-based to 0-based
    label = tf.cast(label, tf.int64) # Ensure consistent type

    # --- Convert to PyTorch Tensors ---
    # Note: tf.py_function or .numpy() is used inside tf.data pipeline.
    # The final output will be converted to PyTorch tensors in the wrapper.

    return image, label

def load_tfrecord_dataset(file_pattern, subset='train', shuffle_buffer_size=1024):
    """Creates a tf.data.Dataset from TFRecord files matching the pattern."""
    # List files matching the pattern
    files = tf.data.Dataset.list_files(file_pattern, shuffle=True) # Shuffle file order

    # Interleave reading TFRecord files for better performance
    dataset = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=4, # Number of files to read in parallel
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False # Allow non-deterministic ordering for speed
    )

    # Parse each example
    dataset = dataset.map(
        lambda x: _parse_tfrecord_fn(x, subset),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Optional: Shuffle examples within a buffer
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Prefetch for performance
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

# --- PyTorch IterableDataset Wrapper ---

class TFRecordImageFolderDataset(IterableDataset):
    """Iterable PyTorch Dataset wrapping a tf.data.Dataset for ImageNet TFRecords."""

    def __init__(self, data_dir, transform=None, shuffle_buffer_size=1024):
        """
        Args:
            data_dir (str or Path): Path to the dataset root (containing /train, /val).
                                    E.g., '/kaggle/input/imagenet-1k-tfrecords-ilsvrc2012-part-0'
            transform (callable, optional): A function/transform that takes in a PIL Image
                                            or tensor and returns a transformed version.
                                            Applied after converting TF tensors to PIL/Tensor.
            shuffle_buffer_size (int): Buffer size for shuffling within tf.data.
                                       Set to 0 to disable shuffling.
        """
        self.data_dir = Path(data_dir)
        # Assume training data is in a 'train' subdirectory
        self.train_dir = self.data_dir / "train"
        if not self.train_dir.exists():
             raise FileNotFoundError(f"Training directory not found: {self.train_dir}")

        self.transform = transform
        self.shuffle_buffer_size = shuffle_buffer_size

        # --- Discover TFRecord files for training ---
        # This pattern matches files within class subdirectories.
        # Adjust the pattern if your files have a specific extension like *.tfrecord
        self.file_pattern = str(self.train_dir / "*/*") # e.g., /path/train/n01440764/...

        # Check if any files match the pattern
        # Note: tf.data.Dataset.list_files can handle empty lists, but good to check.
        # temp_ds = tf.data.Dataset.list_files(self.file_pattern, shuffle=False)
        # try:
        #     first_file = next(iter(temp_ds.take(1)))
        #     print(f"Found TFRecord files. First file example: {first_file}")
        # except StopIteration:
        #     raise FileNotFoundError(f"No TFRecord files found matching pattern: {self.file_pattern}")

    def __iter__(self):
        """Returns an iterator over the dataset."""
        # Get worker info for sharding (important for multi-worker DataLoader)
        worker_info = torch.utils.data.get_worker_info()
        is_main_worker = worker_info is None
        worker_id = 0 if is_main_worker else worker_info.id
        num_workers = 1 if is_main_worker else worker_info.num_workers

        # Create the base tf.data.Dataset
        tf_dataset = load_tfrecord_dataset(
            self.file_pattern,
            subset='train',
            shuffle_buffer_size=self.shuffle_buffer_size
        )

        # --- Sharding for Multi-Worker DataLoader ---
        # tf.data handles sharding effectively.
        # We can shard the file list or the dataset itself.
        # Sharding the dataset after file listing is often simpler.
        if not is_main_worker:
            # Shard the dataset across DataLoader workers
            tf_dataset = tf_dataset.shard(num_workers, worker_id)

        # --- Iterate and Yield PyTorch Samples ---
        # Use tf.data's iterator to fetch examples
        # .as_numpy_iterator() converts tensors to NumPy arrays, which are easier to handle
        for image_tf, label_tf in tf_dataset.as_numpy_iterator():
            try:
                # Convert NumPy arrays (from TF) to PyTorch tensors
                # TF HWC -> PyTorch CHW if needed by transforms
                # The Kaggle example uses HWC, torchvision often expects CHW
                image_tensor = torch.from_numpy(image_tf).permute(2, 0, 1) # HWC to CHW
                label_tensor = torch.tensor(label_tf, dtype=torch.long)

                # Apply transforms
                if self.transform:
                    # Many torchvision transforms work on PIL Images or Tensors
                    # If your transforms expect PIL, convert:
                    # image_pil = Image.fromarray(image_tf.astype('uint8')) # If image_tf was uint8 [0,255]
                    # x1 = self.transform(image_pil)
                    # x2 = self.transform_prime(image_pil) # You'll need to handle the pair logic

                    # If your transforms (like the provided aug.TrainTransform) work on tensors:
                    # You might need to adapt aug.TrainTransform to apply both transforms
                    # and return the pair (x1, x2). Let's assume transform can handle this
                    # or you modify it accordingly.
                    # For now, let's assume `self.transform` returns (x1, x2)
                    x1, x2 = self.transform(image_tensor) # Apply your custom transform logic
                    yield (x1, x2), label_tensor # Yield the pair and the label
                else:
                    # If no transform, just yield the raw tensor and label
                    yield image_tensor, label_tensor

            except Exception as e:
                # Handle potential errors during decoding/processing (e.g., corrupted image)
                print(f"Warning: Skipping example due to error: {e}")
                continue # Skip this example and continue with the next one
