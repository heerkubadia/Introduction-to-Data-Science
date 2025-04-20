"""
Comparison of Standard Bloom Filter and Partitioned Learned Bloom Filter

This script compares the performance of:
1. Standard Bloom Filter (from improved_bloom_testing.py)
2. Partitioned Learned Bloom Filter (PLBF)

Metrics compared:
- False positive rate
- Memory usage
- Processing time
- Bits per item
"""

import pandas as pd
import numpy as np
from pybloom_live import BloomFilter, ScalableBloomFilter
import time
import sys
from sklearn.model_selection import train_test_split
import psutil
import os
from tqdm import tqdm
import gc
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve

# PyTorch imports for GPU support
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class PyTorchLogisticRegression(nn.Module):
    """
    PyTorch implementation of Logistic Regression that can use GPU if available.
    """
    def __init__(self, input_dim):
        super(PyTorchLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out

class PyTorchRandomForest(nn.Module):
    """
    PyTorch implementation of a neural network that approximates a Random Forest.
    This is a simplified version with more hidden units to capture complex patterns.
    """
    def __init__(self, input_dim):
        super(PyTorchRandomForest, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def preprocess_url(url):
    """Extract features from URL for ML model."""
    # Basic preprocessing - convert to lowercase and remove punctuation
    url = url.lower()
    url = re.sub(r'[^\w\s]', '', url)

    # Extract simple features
    features = {
        'length': len(url),
        'num_digits': sum(c.isdigit() for c in url),
        'num_segments': len(url.split('/')),
    }

    return url, features

class CascadedLearnedBloomFilter:
    """
    Cascaded Learned Bloom Filter implementation.

    This filter uses a cascade of machine learning models and Bloom filters
    to achieve optimal model-filter size balance and fast rejection.
    """

    def __init__(self, capacity, error_rate=0.01, num_stages=2, thresholds=None):
        """
        Initialize the Cascaded Learned Bloom Filter.

        Args:
            capacity: Expected number of elements
            error_rate: Desired false positive rate
            num_stages: Number of cascade stages
            thresholds: List of thresholds for each stage (if None, will be determined automatically)
        """
        self.capacity = capacity
        self.error_rate = error_rate
        self.num_stages = num_stages

        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize ML models for each stage
        self.models = []
        self.pytorch_models = []
        self.vectorizers = []
        for _ in range(num_stages):
            self.vectorizers.append(TfidfVectorizer(analyzer='char', ngram_range=(3, 5)))
            self.models.append(None)
            self.pytorch_models.append(None)

        # Thresholds for each stage
        if thresholds is None:
            # Default thresholds will be determined during training
            self.thresholds = [0.5] * num_stages
        else:
            self.thresholds = thresholds

        # Initialize Bloom filters for each stage
        self.bloom_filters = []

        # Statistics
        self.items_added = 0
        self.stage_stats = [{
            'items_processed': 0,
            'items_rejected': 0,
            'items_passed': 0
        } for _ in range(num_stages)]

    def _create_model(self, stage):
        """Create a model for a specific stage."""
        if self.device.type == 'cpu':
            # CPU-based models using scikit-learn
            if stage == 0:
                # First stage uses a simpler model for faster processing
                return Pipeline([
                    ('vectorizer', self.vectorizers[stage]),
                    ('classifier', LogisticRegression(max_iter=1000))
                ])
            else:
                # Later stages use more complex models for better accuracy
                return Pipeline([
                    ('vectorizer', self.vectorizers[stage]),
                    ('classifier', RandomForestClassifier(n_estimators=100))
                ])
        else:
            # GPU-based models using PyTorch
            # We'll create the actual PyTorch models during training
            # This method just returns a placeholder for scikit-learn compatibility
            if stage == 0:
                # First stage uses a simpler model for faster processing
                return Pipeline([
                    ('vectorizer', self.vectorizers[stage]),
                    ('classifier', LogisticRegression(max_iter=1000))
                ])
            else:
                # Later stages use more complex models for better accuracy
                return Pipeline([
                    ('vectorizer', self.vectorizers[stage]),
                    ('classifier', RandomForestClassifier(n_estimators=100))
                ])

    def train_models(self, train_urls, train_labels):
        """
        Train the cascade of ML models.

        Args:
            train_urls: List of URLs for training
            train_labels: Binary labels (0 for safe, 1 for unsafe)
        """
        print("Training cascade of ML models...")

        # Convert to numpy arrays if they're not already
        train_urls = np.array(train_urls)
        train_labels = np.array(train_labels)

        # Keep track of remaining data for each stage
        remaining_urls = train_urls
        remaining_labels = train_labels

        # Train each stage
        for stage in range(self.num_stages):
            print(f"Training stage {stage+1}/{self.num_stages}...")

            # Create and train the model for this stage
            self.models[stage] = self._create_model(stage)

            # First, fit the vectorizer to transform text data
            print(f"Fitting vectorizer for stage {stage+1}...")
            X = self.vectorizers[stage].fit_transform(remaining_urls)

            if self.device.type == 'cpu':
                # CPU training with scikit-learn
                print(f"Fitting model for stage {stage+1} using CPU...")
                # For scikit-learn, we need to fit the classifier part of the pipeline
                self.models[stage].named_steps['classifier'].fit(X, remaining_labels)

                # Get predicted probabilities
                print(f"Getting predicted probabilities for stage {stage+1}...")
                probs = np.zeros(len(remaining_urls))
                progress_bar = tqdm(total=len(remaining_urls), desc=f"Predicting probabilities (stage {stage+1})", unit="url")

                # Process in batches to show progress
                batch_size = 1000
                for i in range(0, len(remaining_urls), batch_size):
                    batch_end = min(i + batch_size, len(remaining_urls))
                    batch_vectors = X[i:batch_end]
                    batch_probs = self.models[stage].named_steps['classifier'].predict_proba(batch_vectors)[:, 1]
                    probs[i:batch_end] = batch_probs
                    progress_bar.update(len(batch_vectors))

                progress_bar.close()
            else:
                # GPU training with PyTorch
                print(f"Fitting model for stage {stage+1} using GPU...")

                # Convert to PyTorch tensors
                X_tensor = torch.FloatTensor(X.toarray()).to(self.device)
                y_tensor = torch.FloatTensor(remaining_labels.reshape(-1, 1)).to(self.device)

                # Create PyTorch model based on stage
                input_dim = X.shape[1]
                if stage == 0:
                    # First stage uses a simpler model for faster processing
                    self.pytorch_models[stage] = PyTorchLogisticRegression(input_dim).to(self.device)
                else:
                    # Later stages use more complex models for better accuracy
                    self.pytorch_models[stage] = PyTorchRandomForest(input_dim).to(self.device)

                # Define loss function and optimizer
                criterion = nn.BCELoss()
                optimizer = optim.Adam(self.pytorch_models[stage].parameters(), lr=0.01)

                # Create DataLoader for batch processing
                dataset = TensorDataset(X_tensor, y_tensor)
                dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

                # Train the model
                print(f"Training PyTorch model for stage {stage+1}...")
                num_epochs = 10
                for epoch in range(num_epochs):
                    running_loss = 0.0
                    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

                    for inputs, labels in progress_bar:
                        # Zero the parameter gradients
                        optimizer.zero_grad()

                        # Forward pass
                        outputs = self.pytorch_models[stage](inputs)
                        loss = criterion(outputs, labels)

                        # Backward pass and optimize
                        loss.backward()
                        optimizer.step()

                        # Update statistics
                        running_loss += loss.item() * inputs.size(0)
                        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                    epoch_loss = running_loss / len(dataset)
                    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

                # Set model to evaluation mode
                self.pytorch_models[stage].eval()

                # Get predicted probabilities
                print(f"Getting predicted probabilities for stage {stage+1}...")
                probs = np.zeros(len(remaining_urls))
                progress_bar = tqdm(total=len(remaining_urls), desc=f"Predicting probabilities (stage {stage+1})", unit="url")

                # Process in batches to show progress
                batch_size = 1000
                with torch.no_grad():
                    for i in range(0, len(remaining_urls), batch_size):
                        batch_end = min(i + batch_size, len(remaining_urls))
                        batch_vectors = X[i:batch_end].toarray()
                        batch_tensor = torch.FloatTensor(batch_vectors).to(self.device)
                        batch_probs = self.pytorch_models[stage](batch_tensor).cpu().numpy().flatten()
                        probs[i:batch_end] = batch_probs
                        progress_bar.update(len(batch_vectors))

                progress_bar.close()

            # Find optimal threshold using ROC curve
            fpr, tpr, thresholds = roc_curve(remaining_labels, probs)

            # Find threshold that gives desired false positive rate
            target_fpr = self.error_rate / self.num_stages  # Distribute error rate across stages
            idx = np.argmin(np.abs(fpr - target_fpr))
            self.thresholds[stage] = thresholds[idx]

            print(f"Optimal threshold for stage {stage+1}: {self.thresholds[stage]:.4f}")

            # Identify items that pass this stage (high probability)
            passed_mask = probs >= self.thresholds[stage]
            passed_urls = remaining_urls[passed_mask]
            passed_labels = remaining_labels[passed_mask]

            # Create Bloom filter for this stage
            if stage < self.num_stages - 1:
                # For intermediate stages, create a Bloom filter for items that don't pass
                rejected_urls = remaining_urls[~passed_mask]
                rejected_labels = remaining_labels[~passed_mask]

                # Only add positive examples (unsafe URLs) to the Bloom filter
                positive_mask = rejected_labels == 1
                positive_urls = rejected_urls[positive_mask]

                # Create a ScalableBloomFilter that can grow as needed
                # This ensures we never run out of capacity
                if len(positive_urls) > 0:
                    stage_error_rate = self.error_rate / self.num_stages
                    # Use a reasonable initial capacity but allow growth
                    initial_capacity = max(len(positive_urls), 100)
                    bloom = ScalableBloomFilter(initial_capacity=initial_capacity, error_rate=stage_error_rate)

                    # Add positive items to Bloom filter
                    print(f"Adding {len(positive_urls)} items to Bloom filter for stage {stage+1}...")
                    progress_bar = tqdm(total=len(positive_urls), desc=f"Adding to Bloom filter (stage {stage+1})", unit="url")

                    for url in positive_urls:
                        bloom.add(url)
                        progress_bar.update(1)

                    progress_bar.close()

                    self.bloom_filters.append(bloom)
                else:
                    # No positive items to add to Bloom filter
                    self.bloom_filters.append(None)
            else:
                # Last stage doesn't need a Bloom filter
                self.bloom_filters.append(None)

            # Update remaining data for next stage
            if stage < self.num_stages - 1:
                remaining_urls = passed_urls
                remaining_labels = passed_labels
                print(f"Items passing to stage {stage+2}: {len(remaining_urls)} ({len(remaining_urls)/len(train_urls)*100:.2f}% of original data)")

                if len(remaining_urls) == 0:
                    print(f"No items left for stage {stage+2}. Stopping cascade training.")
                    break

        print("Cascade training complete.")

    def _get_probability(self, url, stage):
        """
        Get the probability from the model for a single URL at a specific stage.

        Args:
            url: URL to get probability for
            stage: Stage index

        Returns:
            Probability value
        """
        if self.device.type == 'cpu' or self.pytorch_models[stage] is None:
            # Use scikit-learn model
            url_vector = self.vectorizers[stage].transform([url])
            return self.models[stage].named_steps['classifier'].predict_proba(url_vector)[:, 1][0]
        else:
            # Use PyTorch model
            url_vector = self.vectorizers[stage].transform([url]).toarray()
            url_tensor = torch.FloatTensor(url_vector).to(self.device)

            # Get prediction
            with torch.no_grad():
                prob = self.pytorch_models[stage](url_tensor).cpu().numpy()[0, 0]

            return prob

    def add(self, url):
        """
        Add a URL to the filter.

        Args:
            url: URL to add
        """
        if (self.device.type == 'cpu' and not self.models[0]) or \
           (self.device.type == 'cuda' and not self.pytorch_models[0]):
            raise ValueError("Models must be trained before adding elements")

        # Process through the cascade
        for stage in range(self.num_stages):
            # Get probability from model
            prob = self._get_probability(url, stage)

            # Update statistics
            self.stage_stats[stage]['items_processed'] += 1

            if prob < self.thresholds[stage]:
                # Item rejected by this stage's model
                self.stage_stats[stage]['items_rejected'] += 1

                # Add to Bloom filter if this is a positive example (unsafe URL)
                if stage < self.num_stages - 1 and self.bloom_filters[stage]:
                    self.bloom_filters[stage].add(url)

                break
            else:
                # Item passed this stage
                self.stage_stats[stage]['items_passed'] += 1

                # If this is the last stage, we're done
                if stage == self.num_stages - 1:
                    break

        # Update total items added
        self.items_added += 1

    def query(self, url):
        """
        Query if a URL is in the filter.

        Args:
            url: URL to query

        Returns:
            True if the URL might be in the filter, False if definitely not
        """
        if (self.device.type == 'cpu' and not self.models[0]) or \
           (self.device.type == 'cuda' and not self.pytorch_models[0]):
            raise ValueError("Models must be trained before querying")

        # Process through the cascade
        for stage in range(self.num_stages):
            # Get probability from model
            prob = self._get_probability(url, stage)

            if prob < self.thresholds[stage]:
                # Item rejected by this stage's model

                # Check Bloom filter for this stage
                if stage < self.num_stages - 1 and self.bloom_filters[stage]:
                    return url in self.bloom_filters[stage]
                else:
                    # No Bloom filter or last stage, so definitely not in the filter
                    return False
            else:
                # Item passed this stage, continue to next stage
                if stage == self.num_stages - 1:
                    # Last stage passed, so item is in the filter
                    return True

        # Should never reach here
        return False

    def get_stats(self):
        """Return statistics about the filter."""
        # Calculate total bits used by all Bloom filters
        total_bits = 0
        bloom_filter_sizes = []

        for bf in self.bloom_filters:
            if bf:
                # ScalableBloomFilter doesn't have a bitarray attribute
                # Instead, it has a series of BloomFilter objects in its 'filters' attribute
                if hasattr(bf, 'filters'):
                    # For ScalableBloomFilter
                    bf_bits = sum(len(f.bitarray) for f in bf.filters)
                else:
                    # For regular BloomFilter (fallback)
                    bf_bits = len(bf.bitarray)

                total_bits += bf_bits
                bloom_filter_sizes.append(bf_bits / 8 / 1024 / 1024)  # Convert to MB
            else:
                bloom_filter_sizes.append(0)

        total_size_mb = total_bits / 8 / 1024 / 1024  # Convert bits to MB

        # Calculate bits per item
        bits_per_item = total_bits / self.items_added if self.items_added > 0 else 0

        stats = {
            "total_items": self.items_added,
            "num_stages": self.num_stages,
            "thresholds": self.thresholds,
            "stage_stats": self.stage_stats,
            "error_rate": self.error_rate,
            "total_bits": total_bits,
            "total_size_mb": total_size_mb,
            "bits_per_item": bits_per_item,
            "bloom_filter_sizes_mb": bloom_filter_sizes
        }
        return stats

class PartitionedLearnedBloomFilter:
    """
    Partitioned Learned Bloom Filter implementation.

    This filter uses a machine learning model to partition the input space
    and assigns different Bloom filters to different partitions.
    """

    def __init__(self, capacity, error_rate=0.01, num_partitions=3):
        """
        Initialize the Partitioned Learned Bloom Filter.

        Args:
            capacity: Expected number of elements
            error_rate: Desired false positive rate
            num_partitions: Number of partitions to create
        """
        self.capacity = capacity
        self.error_rate = error_rate
        self.num_partitions = num_partitions

        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize ML model
        self.model = None
        self.pytorch_model = None
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))

        # Thresholds for partitioning
        self.thresholds = []

        # Initialize Bloom filters for each partition
        # We'll adjust capacities based on expected elements per partition
        self.bloom_filters = []
        self.partition_capacities = []

        # Statistics
        self.items_added = 0
        self.items_per_partition = [0] * num_partitions

    def train_model(self, train_urls, train_labels):
        """
        Train the ML model for partitioning.

        Args:
            train_urls: List of URLs for training
            train_labels: Binary labels (0 for safe, 1 for unsafe)
        """
        print("Training ML model for partitioning...")

        # First, fit the vectorizer to transform text data
        print("Fitting vectorizer...")
        X = self.vectorizer.fit_transform(train_urls)

        # Convert labels to numpy array if they're not already
        train_labels = np.array(train_labels)

        # For CPU fallback, use scikit-learn pipeline
        if self.device.type == 'cpu':
            print("Using CPU with scikit-learn pipeline")
            self.model = Pipeline([
                ('classifier', LogisticRegression(max_iter=1000))
            ])

            print("Fitting model to training data...")
            self.model.fit(X, train_labels)

            # Get predicted probabilities
            print("Getting predicted probabilities...")
            # Create a progress bar for prediction
            probs = np.zeros(len(train_urls))
            progress_bar = tqdm(total=len(train_urls), desc="Predicting probabilities", unit="url")

            # Process in batches to show progress
            batch_size = 1000
            for i in range(0, len(train_urls), batch_size):
                batch_end = min(i + batch_size, len(train_urls))
                batch_vectors = X[i:batch_end]
                batch_probs = self.model.predict_proba(batch_vectors)[:, 1]
                probs[i:batch_end] = batch_probs
                progress_bar.update(len(batch_vectors))

            progress_bar.close()
        else:
            # For GPU, use PyTorch model
            print("Using GPU with PyTorch model")

            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X.toarray()).to(self.device)
            y_tensor = torch.FloatTensor(train_labels.reshape(-1, 1)).to(self.device)

            # Create PyTorch model
            input_dim = X.shape[1]
            self.pytorch_model = PyTorchLogisticRegression(input_dim).to(self.device)

            # Define loss function and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.pytorch_model.parameters(), lr=0.01)

            # Create DataLoader for batch processing
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

            # Train the model
            print("Training PyTorch model...")
            num_epochs = 10
            for epoch in range(num_epochs):
                running_loss = 0.0
                progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

                for inputs, labels in progress_bar:
                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = self.pytorch_model(inputs)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    # Update statistics
                    running_loss += loss.item() * inputs.size(0)
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                epoch_loss = running_loss / len(dataset)
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

            # Set model to evaluation mode
            self.pytorch_model.eval()

            # Get predicted probabilities
            print("Getting predicted probabilities...")
            probs = np.zeros(len(train_urls))
            progress_bar = tqdm(total=len(train_urls), desc="Predicting probabilities", unit="url")

            # Process in batches to show progress
            batch_size = 1000
            with torch.no_grad():
                for i in range(0, len(train_urls), batch_size):
                    batch_end = min(i + batch_size, len(train_urls))
                    batch_vectors = X[i:batch_end].toarray()
                    batch_tensor = torch.FloatTensor(batch_vectors).to(self.device)
                    batch_probs = self.pytorch_model(batch_tensor).cpu().numpy().flatten()
                    probs[i:batch_end] = batch_probs
                    progress_bar.update(len(batch_vectors))

            progress_bar.close()

        # Compute thresholds for partitioning
        self.thresholds = []
        for i in range(1, self.num_partitions):
            threshold = np.percentile(probs, (i * 100) / self.num_partitions)
            self.thresholds.append(threshold)

        # Count expected elements per partition
        partition_counts = [0] * self.num_partitions
        progress_bar = tqdm(total=len(probs), desc="Calculating partition counts", unit="item")

        # Process in batches for better performance
        batch_size = 10000
        for i in range(0, len(probs), batch_size):
            batch_end = min(i + batch_size, len(probs))
            for prob in probs[i:batch_end]:
                partition = self._get_partition(prob)
                partition_counts[partition] += 1
            progress_bar.update(batch_end - i)

        progress_bar.close()

        # Calculate capacity for each partition
        # Add some buffer to each partition to avoid 100% fill rates
        total_count = sum(partition_counts)
        self.partition_capacities = [
            max(int((count / total_count) * self.capacity * 1.2), 1)  # Add 20% buffer
            for count in partition_counts
        ]

        # Initialize ScalableBloomFilters with adjusted initial capacities
        self.bloom_filters = [
            ScalableBloomFilter(initial_capacity=cap, error_rate=self.error_rate)
            for cap in self.partition_capacities
        ]

        print(f"Model trained. Partition capacities: {self.partition_capacities}")

    def _get_partition(self, probability):
        """
        Determine which partition an element belongs to based on its probability.

        Args:
            probability: ML model's predicted probability

        Returns:
            Partition index (0 to num_partitions-1)
        """
        for i, threshold in enumerate(self.thresholds):
            if probability < threshold:
                return i
        return self.num_partitions - 1

    def _get_probability(self, url):
        """
        Get the probability from the model for a single URL.

        Args:
            url: URL to get probability for

        Returns:
            Probability value
        """
        if self.device.type == 'cpu' or self.pytorch_model is None:
            # Use scikit-learn model
            url_vector = self.vectorizer.transform([url])
            return self.model.predict_proba(url_vector)[0, 1]
        else:
            # Use PyTorch model
            url_vector = self.vectorizer.transform([url]).toarray()
            url_tensor = torch.FloatTensor(url_vector).to(self.device)

            # Get prediction
            with torch.no_grad():
                prob = self.pytorch_model(url_tensor).cpu().numpy()[0, 0]

            return prob

    def add(self, url):
        """
        Add a URL to the filter.

        Args:
            url: URL to add
        """
        if (self.device.type == 'cpu' and self.model is None) or \
           (self.device.type == 'cuda' and self.pytorch_model is None):
            raise ValueError("Model must be trained before adding elements")

        # Get probability from model
        prob = self._get_probability(url)

        # Determine partition
        partition = self._get_partition(prob)

        # Add to corresponding Bloom filter
        self.bloom_filters[partition].add(url)

        # Update statistics
        self.items_added += 1
        self.items_per_partition[partition] += 1

    def query(self, url):
        """
        Query if a URL is in the filter.

        Args:
            url: URL to query

        Returns:
            True if the URL might be in the filter, False if definitely not
        """
        if (self.device.type == 'cpu' and self.model is None) or \
           (self.device.type == 'cuda' and self.pytorch_model is None):
            raise ValueError("Model must be trained before querying")

        # Get probability from model
        prob = self._get_probability(url)

        # Determine partition
        partition = self._get_partition(prob)

        # Query corresponding Bloom filter
        return url in self.bloom_filters[partition]

    def get_stats(self):
        """Return statistics about the filter."""
        # Calculate total bits used by all Bloom filters
        total_bits = 0
        bloom_filter_sizes_mb = []

        for bf in self.bloom_filters:
            # ScalableBloomFilter doesn't have a bitarray attribute
            # Instead, it has a series of BloomFilter objects in its 'filters' attribute
            if hasattr(bf, 'filters'):
                # For ScalableBloomFilter
                bf_bits = sum(len(f.bitarray) for f in bf.filters)
                bf_size_mb = bf_bits / 8 / 1024 / 1024  # Convert to MB
            else:
                # For regular BloomFilter (fallback)
                bf_bits = len(bf.bitarray)
                bf_size_mb = bf_bits / 8 / 1024 / 1024  # Convert to MB

            total_bits += bf_bits
            bloom_filter_sizes_mb.append(bf_size_mb)

        total_size_mb = total_bits / 8 / 1024 / 1024  # Convert bits to MB

        # Calculate bits per item
        bits_per_item = total_bits / self.items_added if self.items_added > 0 else 0

        stats = {
            "total_items": self.items_added,
            "items_per_partition": self.items_per_partition,
            "partition_capacities": self.partition_capacities,
            "partition_fill_rates": [
                self.items_per_partition[i] / self.partition_capacities[i]
                for i in range(self.num_partitions)
            ],
            "num_partitions": self.num_partitions,
            "error_rate": self.error_rate,
            "total_bits": total_bits,
            "total_size_mb": total_size_mb,
            "bits_per_item": bits_per_item,
            "bloom_filter_sizes_mb": bloom_filter_sizes_mb
        }
        return stats

def get_memory_usage():
    """Return the current memory usage in MB."""
    # Force garbage collection before measuring memory
    gc.collect()
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def evaluate_standard_bloom_filter(train_urls, test_urls, capacity, error_rate):
    """
    Evaluate a standard Bloom filter.

    Args:
        train_urls: URLs to add to the filter
        test_urls: URLs to test against the filter
        capacity: Expected number of elements
        error_rate: Desired false positive rate

    Returns:
        Dictionary with evaluation results
    """
    print(f"\nEvaluating Standard Bloom Filter with error_rate={error_rate}")

    # Initialize memory and time tracking
    start_time = time.time()
    memory_before = get_memory_usage()
    print(f"Initial memory usage: {memory_before:.2f} MB")

    # Create Bloom filter
    bloom = BloomFilter(capacity=capacity, error_rate=error_rate)

    # Calculate filter size in bits and MB
    filter_bits = len(bloom.bitarray)
    filter_size_mb = filter_bits / 8 / 1024 / 1024

    # Track memory after filter creation
    memory_after_creation = get_memory_usage()
    creation_time = time.time() - start_time

    # Add all training URLs to the filter
    print(f"Adding {len(train_urls)} URLs to the filter...")
    add_start_time = time.time()
    progress_bar = tqdm(total=len(train_urls), desc="Adding URLs", unit="url")

    # Process in batches for better progress tracking
    batch_size = 1000
    for i in range(0, len(train_urls), batch_size):
        batch_end = min(i + batch_size, len(train_urls))
        for url in train_urls[i:batch_end]:
            bloom.add(url)
        progress_bar.update(batch_end - i)

    progress_bar.close()

    add_time = time.time() - add_start_time
    memory_after_add = get_memory_usage()
    print(f"Memory after adding URLs: {memory_after_add:.2f} MB")

    # Test for true positives (URLs from training set)
    print("Testing for true positives (URLs from training set)...")
    tp_start_time = time.time()
    true_positives = 0
    false_negatives = 0

    # Sample a subset of training URLs for testing (to save time)
    sample_size = min(10000, len(train_urls))
    sample_indices = np.random.choice(len(train_urls), sample_size, replace=False)

    progress_bar = tqdm(total=len(sample_indices), desc="Testing true positives", unit="url")

    # Process in batches
    batch_size = 500
    for batch_start in range(0, len(sample_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(sample_indices))
        for i in sample_indices[batch_start:batch_end]:
            if train_urls[i] in bloom:
                true_positives += 1
            else:
                false_negatives += 1
        progress_bar.update(batch_end - batch_start)

    progress_bar.close()

    tp_time = time.time() - tp_start_time

    # Test for false positives (URLs from test set)
    print("Testing for false positives (URLs from test set)...")
    fp_start_time = time.time()
    true_negatives = 0
    false_positives = 0

    progress_bar = tqdm(total=len(test_urls), desc="Testing false positives", unit="url")

    # Process in batches
    batch_size = 1000
    for i in range(0, len(test_urls), batch_size):
        batch_end = min(i + batch_size, len(test_urls))
        for url in test_urls[i:batch_end]:
            if url in bloom:
                false_positives += 1
            else:
                true_negatives += 1
        progress_bar.update(batch_end - i)

    progress_bar.close()

    fp_time = time.time() - fp_start_time

    # Calculate metrics
    false_positive_rate = false_positives / len(test_urls) if len(test_urls) > 0 else 0
    false_negative_rate = false_negatives / sample_size if sample_size > 0 else 0
    bits_per_item = filter_bits / len(train_urls) if len(train_urls) > 0 else 0

    # Prepare results
    results = {
        "filter_type": f"Standard Bloom Filter (error_rate={error_rate})",
        "error_rate": error_rate,
        "capacity": capacity,
        "memory_before": memory_before,
        "memory_after_creation": memory_after_creation,
        "memory_after_add": memory_after_add,
        "memory_for_filter": memory_after_add - memory_before,
        "filter_size_bits": filter_bits,
        "filter_size_mb": filter_size_mb,
        "bits_per_item": bits_per_item,
        "creation_time": creation_time,
        "add_time": add_time,
        "test_time_positives": tp_time,
        "test_time_negatives": fp_time,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }

    # Print summary
    print("\nResults:")
    print(f"  False Positive Rate: {false_positive_rate:.4f} (expected: {error_rate})")
    print(f"  False Negative Rate: {false_negative_rate:.4f}")
    print(f"  Filter Size: {filter_size_mb:.2f} MB ({filter_bits} bits)")
    print(f"  Bits per item: {bits_per_item:.2f}")
    print(f"  Memory usage: {results['memory_for_filter']:.2f} MB")
    print(f"  Add time: {add_time:.2f} seconds")
    print(f"  Test time: {tp_time + fp_time:.2f} seconds")

    return results

def evaluate_partitioned_learned_bloom_filter(train_urls, train_labels, test_urls, test_labels, capacity, error_rate, num_partitions):
    """
    Evaluate a Partitioned Learned Bloom Filter.

    Args:
        train_urls: URLs for training
        train_labels: Labels for training (0 for safe, 1 for unsafe)
        test_urls: URLs for testing
        test_labels: Labels for testing
        capacity: Expected number of elements
        error_rate: Desired false positive rate
        num_partitions: Number of partitions

    Returns:
        Dictionary with evaluation results
    """
    print(f"\nEvaluating Partitioned Learned Bloom Filter with {num_partitions} partitions and error_rate={error_rate}")

    # Initialize memory and time tracking
    start_time = time.time()
    memory_before = get_memory_usage()
    print(f"Initial memory usage: {memory_before:.2f} MB")

    # Create and train PLBF
    plbf = PartitionedLearnedBloomFilter(capacity, error_rate, num_partitions)
    plbf.train_model(train_urls, train_labels)

    # Track memory after model creation
    memory_after_model = get_memory_usage()
    print(f"Memory after model creation: {memory_after_model:.2f} MB")
    model_creation_time = time.time() - start_time

    # Add all training URLs to the filter
    print(f"Adding {len(train_urls)} URLs to the filter...")
    add_start_time = time.time()
    progress_bar = tqdm(total=len(train_urls), desc="Adding URLs", unit="url")

    # Process in batches for better progress tracking
    batch_size = 1000
    for i in range(0, len(train_urls), batch_size):
        batch_end = min(i + batch_size, len(train_urls))
        for url in train_urls[i:batch_end]:
            plbf.add(url)
        progress_bar.update(batch_end - i)

    progress_bar.close()

    add_time = time.time() - add_start_time
    memory_after_add = get_memory_usage()
    print(f"Memory after adding URLs: {memory_after_add:.2f} MB")

    # Test for true positives (URLs from training set)
    print("Testing for true positives (URLs from training set)...")
    tp_start_time = time.time()
    true_positives = 0
    false_negatives = 0

    # Sample a subset of training URLs for testing (to save time)
    sample_size = min(10000, len(train_urls))
    sample_indices = np.random.choice(len(train_urls), sample_size, replace=False)

    progress_bar = tqdm(total=len(sample_indices), desc="Testing true positives", unit="url")

    # Process in batches
    batch_size = 500
    for batch_start in range(0, len(sample_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(sample_indices))
        for i in sample_indices[batch_start:batch_end]:
            if plbf.query(train_urls[i]):
                true_positives += 1
            else:
                false_negatives += 1
        progress_bar.update(batch_end - batch_start)

    progress_bar.close()

    tp_time = time.time() - tp_start_time

    # Test for false positives (URLs from test set)
    print("Testing for false positives (URLs from test set)...")
    fp_start_time = time.time()
    true_negatives = 0
    false_positives = 0

    progress_bar = tqdm(total=len(test_urls), desc="Testing false positives", unit="url")

    # Process in batches
    batch_size = 1000
    for i in range(0, len(test_urls), batch_size):
        batch_end = min(i + batch_size, len(test_urls))
        for url in test_urls[i:batch_end]:
            if plbf.query(url):
                false_positives += 1
            else:
                true_negatives += 1
        progress_bar.update(batch_end - i)

    progress_bar.close()

    fp_time = time.time() - fp_start_time

    # Calculate metrics
    false_positive_rate = false_positives / len(test_urls) if len(test_urls) > 0 else 0
    false_negative_rate = false_negatives / sample_size if sample_size > 0 else 0

    # Get filter stats
    filter_stats = plbf.get_stats()

    # Prepare results
    results = {
        "filter_type": f"PLBF (partitions={num_partitions}, error_rate={error_rate})",
        "num_partitions": num_partitions,
        "error_rate": error_rate,
        "capacity": capacity,
        "memory_before": memory_before,
        "memory_after_model": memory_after_model,
        "memory_after_add": memory_after_add,
        "memory_for_model": memory_after_model - memory_before,
        "memory_for_filters": filter_stats["total_size_mb"],
        "total_memory": (memory_after_model - memory_before) + filter_stats["total_size_mb"],
        "model_creation_time": model_creation_time,
        "add_time": add_time,
        "test_time_positives": tp_time,
        "test_time_negatives": fp_time,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "items_per_partition": filter_stats["items_per_partition"],
        "partition_fill_rates": filter_stats["partition_fill_rates"],
        "bits_per_item": filter_stats["bits_per_item"],
        "total_bits": filter_stats["total_bits"],
    }

    # Print summary
    print("\nResults:")
    print(f"  False Positive Rate: {false_positive_rate:.4f} (expected: {error_rate})")
    print(f"  False Negative Rate: {false_negative_rate:.4f}")
    print(f"  Memory for model: {results['memory_for_model']:.2f} MB")
    print(f"  Memory for Bloom filters: {results['memory_for_filters']:.2f} MB")
    print(f"  Total memory (model + filters): {results['total_memory']:.2f} MB")
    print(f"  Bits per item: {results['bits_per_item']:.2f}")
    print(f"  Model creation time: {model_creation_time:.2f} seconds")
    print(f"  Add time: {add_time:.2f} seconds")
    print(f"  Test time: {tp_time + fp_time:.2f} seconds")
    print(f"  Items per partition: {filter_stats['items_per_partition']}")
    print(f"  Partition fill rates: {[f'{rate:.2f}' for rate in filter_stats['partition_fill_rates']]}")

    return results

def evaluate_cascaded_learned_bloom_filter(train_urls, train_labels, test_urls, test_labels, capacity, error_rate=0.01, num_stages=2):
    """
    Evaluate a Cascaded Learned Bloom Filter.

    Args:
        train_urls: URLs for training
        train_labels: Labels for training (0 for safe, 1 for unsafe)
        test_urls: URLs for testing
        test_labels: Labels for testing (not used in current implementation but included for future label-specific analysis)
        capacity: Expected number of elements
        error_rate: Desired false positive rate
        num_stages: Number of cascade stages

    Returns:
        Dictionary with evaluation results
    """
    print(f"\nEvaluating Cascaded Learned Bloom Filter with {num_stages} stages and error_rate={error_rate}")

    # Initialize memory and time tracking
    start_time = time.time()
    memory_before = get_memory_usage()
    print(f"Initial memory usage: {memory_before:.2f} MB")

    # Create and train CLBF
    clbf = CascadedLearnedBloomFilter(capacity, error_rate, num_stages)
    clbf.train_models(train_urls, train_labels)

    # Track memory after model creation
    memory_after_model = get_memory_usage()
    print(f"Memory after model creation: {memory_after_model:.2f} MB")
    model_creation_time = time.time() - start_time

    # Add all training URLs to the filter
    print(f"Adding {len(train_urls)} URLs to the filter...")
    add_start_time = time.time()
    progress_bar = tqdm(total=len(train_urls), desc="Adding URLs", unit="url")

    # Process in batches for better progress tracking
    batch_size = 1000
    for i in range(0, len(train_urls), batch_size):
        batch_end = min(i + batch_size, len(train_urls))
        for url in train_urls[i:batch_end]:
            clbf.add(url)
        progress_bar.update(batch_end - i)

    progress_bar.close()

    add_time = time.time() - add_start_time
    memory_after_add = get_memory_usage()
    print(f"Memory after adding URLs: {memory_after_add:.2f} MB")

    # Test for true positives (URLs from training set)
    print("Testing for true positives (URLs from training set)...")
    tp_start_time = time.time()
    true_positives = 0
    false_negatives = 0

    # Sample a subset of training URLs for testing (to save time)
    sample_size = min(10000, len(train_urls))
    sample_indices = np.random.choice(len(train_urls), sample_size, replace=False)

    progress_bar = tqdm(total=len(sample_indices), desc="Testing true positives", unit="url")

    # Process in batches
    batch_size = 500
    for batch_start in range(0, len(sample_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(sample_indices))
        for i in sample_indices[batch_start:batch_end]:
            if clbf.query(train_urls[i]):
                true_positives += 1
            else:
                false_negatives += 1
        progress_bar.update(batch_end - batch_start)

    progress_bar.close()

    tp_time = time.time() - tp_start_time

    # Test for false positives (URLs from test set)
    print("Testing for false positives (URLs from test set)...")
    fp_start_time = time.time()
    true_negatives = 0
    false_positives = 0

    progress_bar = tqdm(total=len(test_urls), desc="Testing false positives", unit="url")

    # Process in batches
    batch_size = 1000
    for i in range(0, len(test_urls), batch_size):
        batch_end = min(i + batch_size, len(test_urls))
        for url in test_urls[i:batch_end]:
            if clbf.query(url):
                false_positives += 1
            else:
                true_negatives += 1
        progress_bar.update(batch_end - i)

    progress_bar.close()

    fp_time = time.time() - fp_start_time

    # Calculate metrics
    false_positive_rate = false_positives / len(test_urls) if len(test_urls) > 0 else 0
    false_negative_rate = false_negatives / sample_size if sample_size > 0 else 0

    # Get filter stats
    filter_stats = clbf.get_stats()

    # Prepare results
    results = {
        "filter_type": f"CLBF (stages={num_stages}, error_rate={error_rate})",
        "num_stages": num_stages,
        "error_rate": error_rate,
        "capacity": capacity,
        "memory_before": memory_before,
        "memory_after_model": memory_after_model,
        "memory_after_add": memory_after_add,
        "memory_for_model": memory_after_model - memory_before,
        "memory_for_filters": filter_stats["total_size_mb"],
        "total_memory": (memory_after_model - memory_before) + filter_stats["total_size_mb"],
        "model_creation_time": model_creation_time,
        "add_time": add_time,
        "test_time_positives": tp_time,
        "test_time_negatives": fp_time,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "stage_stats": filter_stats["stage_stats"],
        "thresholds": filter_stats["thresholds"],
        "bits_per_item": filter_stats["bits_per_item"],
        "total_bits": filter_stats["total_bits"],
    }

    # Print summary
    print("\nResults:")
    print(f"  False Positive Rate: {false_positive_rate:.4f} (expected: {error_rate})")
    print(f"  False Negative Rate: {false_negative_rate:.4f}")
    print(f"  Memory for model: {results['memory_for_model']:.2f} MB")
    print(f"  Memory for Bloom filters: {results['memory_for_filters']:.2f} MB")
    print(f"  Total memory (model + filters): {results['total_memory']:.2f} MB")
    print(f"  Bits per item: {results['bits_per_item']:.2f}")
    print(f"  Model creation time: {model_creation_time:.2f} seconds")
    print(f"  Add time: {add_time:.2f} seconds")
    print(f"  Test time: {tp_time + fp_time:.2f} seconds")
    print(f"  Stage statistics:")
    for stage in range(num_stages):
        stats = filter_stats["stage_stats"][stage]
        print(f"    Stage {stage+1}: Processed {stats['items_processed']}, "
              f"Rejected {stats['items_rejected']} ({stats['items_rejected']/stats['items_processed']*100 if stats['items_processed'] > 0 else 0:.2f}%), "
              f"Passed {stats['items_passed']} ({stats['items_passed']/stats['items_processed']*100 if stats['items_processed'] > 0 else 0:.2f}%)")
    print(f"  Bloom filter sizes: {[f'{size:.2f} MB' for size in filter_stats['bloom_filter_sizes_mb']]}")

    return results

def compare_results(standard_results, plbf_results, clbf_results):
    """
    Compare and print the results of standard, partitioned, and cascaded Bloom filters.

    Args:
        standard_results: Results from standard Bloom filter evaluation
        plbf_results: Results from PLBF evaluation
        clbf_results: Results from CLBF evaluation
    """
    print("\n" + "="*80)
    print("COMPARISON OF STANDARD BLOOM FILTER VS PARTITIONED LEARNED BLOOM FILTER VS CASCADED LEARNED BLOOM FILTER")
    print("="*80)

    # Compare false positive rates
    print("\nFalse Positive Rates:")
    for result in standard_results:
        print(f"  {result['filter_type']}: {result['false_positive_rate']:.4f} (expected: {result['error_rate']})")
    for result in plbf_results:
        print(f"  {result['filter_type']}: {result['false_positive_rate']:.4f} (expected: {result['error_rate']})")
    for result in clbf_results:
        print(f"  {result['filter_type']}: {result['false_positive_rate']:.4f} (expected: {result['error_rate']})")

    # Compare memory usage
    print("\nMemory Usage:")
    for result in standard_results:
        print(f"  {result['filter_type']}: {result['filter_size_mb']:.2f} MB")
    for result in plbf_results:
        print(f"  {result['filter_type']}: {result['memory_for_model']:.2f} MB (model) + {result['memory_for_filters']:.2f} MB (filters) = {result['total_memory']:.2f} MB (total)")
    for result in clbf_results:
        print(f"  {result['filter_type']}: {result['memory_for_model']:.2f} MB (model) + {result['memory_for_filters']:.2f} MB (filters) = {result['total_memory']:.2f} MB (total)")

    # Compare bits per item
    print("\nBits per Item:")
    for result in standard_results:
        print(f"  {result['filter_type']}: {result['bits_per_item']:.2f}")
    for result in plbf_results:
        print(f"  {result['filter_type']}: {result['bits_per_item']:.2f}")
    for result in clbf_results:
        print(f"  {result['filter_type']}: {result['bits_per_item']:.2f}")

    # Compare add times
    print("\nAdd Times:")
    for result in standard_results:
        print(f"  {result['filter_type']}: {result['add_time']:.2f} seconds")
    for result in plbf_results:
        print(f"  {result['filter_type']}: {result['add_time']:.2f} seconds (+ {result['model_creation_time']:.2f} seconds for model training)")
    for result in clbf_results:
        print(f"  {result['filter_type']}: {result['add_time']:.2f} seconds (+ {result['model_creation_time']:.2f} seconds for model training)")

    # Compare query times
    print("\nQuery Times (for both true positives and false positives):")
    for result in standard_results:
        total_query_time = result['test_time_positives'] + result['test_time_negatives']
        print(f"  {result['filter_type']}: {total_query_time:.2f} seconds")
    for result in plbf_results:
        total_query_time = result['test_time_positives'] + result['test_time_negatives']
        print(f"  {result['filter_type']}: {total_query_time:.2f} seconds")
    for result in clbf_results:
        total_query_time = result['test_time_positives'] + result['test_time_negatives']
        print(f"  {result['filter_type']}: {total_query_time:.2f} seconds")

    print("\nSUMMARY:")
    print("  Standard Bloom Filter advantages:")
    print("    - Simpler implementation")
    print("    - No training required")
    print("    - Faster setup time")
    print("    - Lower memory overhead for small datasets")

    print("\n  Partitioned Learned Bloom Filter advantages:")
    print("    - Can achieve lower false positive rates for the same memory usage")
    print("    - More efficient memory utilization for large datasets")
    print("    - Can be optimized for specific data distributions")
    print("    - Better scalability for very large datasets")

    print("\n  Cascaded Learned Bloom Filter advantages:")
    print("    - Fast rejection of obvious negatives in early stages")
    print("    - Optimal balance between model complexity and filter size")
    print("    - Progressive refinement through multiple stages")
    print("    - Can use different model complexities at different stages")

    print("\nRECOMMENDATION:")
    print("  - For small datasets or simple use cases: Standard Bloom Filter")
    print("  - For large datasets with uniform distribution: Partitioned Learned Bloom Filter")
    print("  - For datasets with varying difficulty of classification: Cascaded Learned Bloom Filter")
    print("  - When query speed is critical: Cascaded Learned Bloom Filter (for fast rejection)")

def main(length_to_consider=None):
    """Main function to run the comparison.

    Args:
        length_to_consider: Optional integer to limit the number of rows to use from the dataset.
                           If None, the entire dataset is used.
    """
    # Check if CUDA is available and print device information
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

    print("Loading dataset...")
    df = pd.read_csv('/kaggle/input/malicious-url/binary_url_dataset.csv')
    print(f"Loaded {len(df)} rows from binary_url_dataset.csv")

    # Limit the dataset size if specified
    if length_to_consider is not None and length_to_consider < len(df):
        print(f"Limiting dataset to {length_to_consider} rows for faster processing")
        # Stratified sampling to maintain label distribution
        df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=min(len(x), length_to_consider // 2)))
        print(f"Dataset after limiting: {len(df)} rows")

    # Check for duplicates
    print("Checking for duplicates...")
    duplicates = df.duplicated(subset=['url']).sum()
    print(f"Found {duplicates} duplicate URLs ({duplicates/len(df)*100:.2f}% of the dataset)")

    # Remove duplicates
    df = df.drop_duplicates(subset=['url'])
    print(f"Dataset after removing duplicates: {len(df)} rows")

    # Convert labels to binary (0 for safe, 1 for unsafe)
    df['binary_label'] = df['label'].apply(lambda x: 1 if x == 'unsafe' else 0)

    # Print label distribution
    safe_count = (df['binary_label'] == 0).sum()
    unsafe_count = (df['binary_label'] == 1).sum()
    print(f"Label distribution: {safe_count} safe URLs, {unsafe_count} unsafe URLs")

    # Split data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['binary_label'], random_state=42)

    print(f"Training set: {len(train_df)} URLs")
    print(f"Testing set: {len(test_df)} URLs")

    # Verify no overlap between training and testing sets
    train_urls = set(train_df['url'].values)
    test_urls = set(test_df['url'].values)
    overlap = train_urls.intersection(test_urls)
    print(f"Overlap between training and testing sets: {len(overlap)} URLs")

    # Extract URLs and labels
    train_urls = train_df['url'].values
    train_labels = train_df['binary_label'].values
    test_urls = test_df['url'].values
    test_labels = test_df['binary_label'].values

    # Set capacity to the number of training URLs
    capacity = len(train_urls)

    # Evaluate standard Bloom filters with different error rates
    standard_results = []
    for error_rate in [0.1, 0.01, 0.001]:
        result = evaluate_standard_bloom_filter(
            train_urls, test_urls, capacity=capacity, error_rate=error_rate
        )
        standard_results.append(result)

    # Evaluate PLBFs with different configurations
    plbf_results = []

    # Test with different error rates
    for error_rate in [0.1, 0.01, 0.001]:
        result = evaluate_partitioned_learned_bloom_filter(
            train_urls, train_labels, test_urls, test_labels,
            capacity=capacity, error_rate=error_rate, num_partitions=3
        )
        plbf_results.append(result)

    # Evaluate CLBFs with different configurations
    clbf_results = []

    # Test with different error rates
    for error_rate in [0.1, 0.01, 0.001]:
        result = evaluate_cascaded_learned_bloom_filter(
            train_urls, train_labels, test_urls, test_labels,
            capacity=capacity, error_rate=error_rate, num_stages=5
        )
        clbf_results.append(result)

    # Compare the results
    compare_results(standard_results, plbf_results, clbf_results)

    print("\nComparison complete!")

if __name__ == "__main__":
    # Set the length_to_consider parameter here
    # Use None to use the entire dataset, or set a specific number (e.g., 10000) for faster processing
    length_to_consider = 30000  # Change this value as needed

    # Run main function with specified length
    main(length_to_consider=length_to_consider)
