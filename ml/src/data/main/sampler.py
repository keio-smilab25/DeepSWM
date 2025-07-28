from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm
import numpy as np
import torch
from src.data.main.dataset import SolarFlareDatasetWithFeatures
from numpy import int64
from typing import Iterator, List
import os
import pickle


class TrainBalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset: SolarFlareDatasetWithFeatures, n_classes: int, n_samples: int, fold_index: int):
        print("Prepare Batch Sampler ...")
        self.dataset = dataset
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = self.n_samples * self.n_classes

        self.cache_dir = os.path.join(
            dataset.args.cache_root, 
            "cache",
            f"fold{fold_index}",
            "sampler"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_file = os.path.join(
            self.cache_dir, 
            f'sampler_cache_{len(dataset)}_{n_classes}_{n_samples}.pkl'
        )

        if os.path.exists(self.cache_file) and not dataset.args.dataset.get("force_recalc_sampler", False):
            self._load_cache()
        else:
            self._prepare_and_cache()

    def _load_cache(self):
        print("Loading sampler cache...")
        with open(self.cache_file, 'rb') as f:
            cache = pickle.load(f)
        self.labels_list = cache['labels_list']
        self.labels = cache['labels']
        self.labels_set = cache['labels_set']
        self.label_to_indices = cache['label_to_indices']

    def _prepare_and_cache(self):
        print("Preparing sampler data...")
        self.labels_list = [torch.argmax(self.dataset[i][2]).item() for i in tqdm(range(len(self.dataset)), desc="Getting labels")]
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0] for label in tqdm(self.labels_set, desc="Creating label_to_indices")}

        for l in tqdm(self.labels_set, desc="Shuffling indices"):
            np.random.shuffle(self.label_to_indices[l])

        cache = {
            'labels_list': self.labels_list,
            'labels': self.labels,
            'labels_set': self.labels_set,
            'label_to_indices': self.label_to_indices
        }

        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache, f)

    def __iter__(self) -> Iterator[List[int64]]:
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                class_indices = self.label_to_indices[class_]
                if self.used_label_indices_count[class_] + self.n_samples > len(class_indices):
                    np.random.shuffle(class_indices)
                    self.used_label_indices_count[class_] = 0
                
                indices.extend(class_indices[self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
            
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size
