"""Scripts for evaluation metrics"""

import math
import torch
from sklearn import metrics
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from numpy import ndarray
import logging
import os
import pandas as pd
import h5py
from datetime import datetime


class Stat:
    """
    collect predictions and calculate some metrics
    """

    def __init__(self, climatology: np.ndarray, logger: logging.Logger):
        self.predictions = {"train": [], "valid": [], "test": []}
        self.observations = {"train": [], "valid": [], "test": []}
        self.climatology = climatology
        self.logger = logger
        self.gmgs_score_matrix = None

    @property
    def gmgs_score_matrix(self):
        return self._gmgs_score_matrix

    @gmgs_score_matrix.setter
    def gmgs_score_matrix(self, matrix):
        self._gmgs_score_matrix = matrix

    def collect(self, pred: torch.Tensor, ground_truth: torch.Tensor, mode: str):
        """
        Collect predictions and ground truth
        """
        assert mode in [
            "train",
            "valid",
            "test",
        ], "Mode must be either 'train', 'valid' or 'test'"
        observation = torch.argmax(ground_truth, dim=1)
        self.predictions[mode].extend(pred.cpu().detach().numpy())
        self.observations[mode].extend(observation.cpu().detach().numpy())

    def aggregate(self, mode: str) -> Dict[str, Any]:
        """
        Aggregate collected data and calculate metrics
        """
        assert mode in [
            "train",
            "valid",
            "test",
        ], "Mode must be either 'train', 'valid' or 'test'"
        score = {}
        y_pred, y_true = np.array(self.predictions[mode]), np.array(
            self.observations[mode]
        )
        y_predl = [np.argmax(y) for y in y_pred]

        score["ACC"] = self.calc_accs(y_predl, y_true)
        score["TSS"] = self.calc_tss(y_predl, y_true, 2)
        score["BSS"] = self.calc_bss(y_pred, y_true, self.climatology)
        score["GMGS"] = self.calc_gmgs(y_predl, y_true)

        results = {f"{mode}_{k}": v for k, v in score.items()}

        return results

    def clear_all(self):
        """
        Clear all collected data
        """
        self.predictions = {"train": [], "valid": [], "test": []}
        self.observations = {"train": [], "valid": [], "test": []}

    def calc_mattheus(self, y_predl: ndarray, y_true: ndarray, flare_class: int):
        """
        Compute Matthews correlation coefficient
        """
        C = self.confusion_matrix(y_predl, y_true)
        c, s = np.diag(C).sum(), C.sum()
        p, t = np.sum(C, axis=0), np.sum(C, axis=1)
        mcc = (c * s - np.dot(p, t).sum()) / np.sqrt(
            (s**2 - np.dot(p, p).sum()) - (s**2 - np.dot(t, t).sum())
        )
        return mcc

    def calc_tss(self, y_predl: ndarray, y_true: ndarray, flare_class: int) -> float:
        """
        Compute TSS
        """
        mtx = self.confusion_matrix(y_predl, y_true)
        tn, fp, fn, tp = self.binary_confusion_matrix(mtx, flare_class)
        print("tn, fp, fn, tp", tn, fp, fn, tp)
        # Add check to avoid division by zero
        if (tp + fn) == 0 or (fp + tn) == 0:
            return 0.0

        tss = (tp / (tp + fn)) - (fp / (fp + tn))
        return float(tss) if not math.isnan(tss) else 0.0

    def calc_gmgs(self, y_predl: ndarray, y_true: ndarray) -> float:
        """
        Compute GMGS according to paper definition: GMGS = tr(S^T * P)
        where S is scoring matrix and P is normalized contingency table
        """
        # Create contingency table (confusion matrix)
        P_raw = self.confusion_matrix(y_predl, y_true).astype(float)
        
        # Normalize contingency table P as required by paper
        total_samples = np.sum(P_raw)
        if total_samples > 0:
            P = P_raw / total_samples  # Normalize so that sum(P) = 1
        else:
            P = P_raw  # Handle edge case
            
        I = P.shape[0]  # Number of classes
        
        # Calculate row sums p_i (equation 16) from normalized P
        p = np.sum(P, axis=1)  # p_i = sum_j(p_ij)
        
        # Handle case where some classes have no samples
        if np.any(p == 0):
            # Add small epsilon to avoid division by zero
            p = p + 1e-8
        
        # Calculate a_i (equation 15)
        a = np.zeros(I)
        for i in range(I):
            sum_pk_1_to_i = np.sum(p[:i+1])
            if sum_pk_1_to_i > 0:
                a[i] = (1 - sum_pk_1_to_i) / sum_pk_1_to_i
            else:
                a[i] = 0.0
        
        # Calculate scoring matrix S
        S = np.zeros((I, I))
        
        # Calculate diagonal elements s_ii (equation 13)
        for i in range(I):
            sum_ak_inv = 0.0
            sum_ak = 0.0
            
            # sum(a_k^-1) for k=1 to i-1
            for k in range(i):
                if a[k] != 0:
                    sum_ak_inv += 1.0 / a[k]
            
            # sum(a_k) for k=i to I-1  
            for k in range(i, I):
                sum_ak += a[k]
            
            if I > 1:
                S[i, i] = (sum_ak_inv + sum_ak) / (I - 1)
        
        # Calculate off-diagonal elements s_ij (equation 14)
        for i in range(I):
            for j in range(i + 1, I):
                sum_ak_inv = 0.0
                sum_ak = 0.0
                count_neg_one = j - i  # number of -1 terms
                
                # sum(a_k^-1) for k=1 to i-1
                for k in range(i):
                    if a[k] != 0:
                        sum_ak_inv += 1.0 / a[k]
                
                # sum(a_k) for k=j to I-1
                for k in range(j, I):
                    sum_ak += a[k]
                
                if I > 1:
                    S[i, j] = (sum_ak_inv - count_neg_one + sum_ak) / (I - 1)
                    S[j, i] = S[i, j]  # Symmetric matrix
        
        # Calculate GMGS = tr(S^T * P) with normalized P
        gmgs = np.trace(np.dot(S.T, P))
        
        return float(gmgs)

    def calc_bss(
        self, y_pred: ndarray, y_true: ndarray, climatology: List[float]
    ) -> float:
        """
        Compute BSS >= M
        """
        N = len(y_true)
        y_truel = np.array([self.convert_binary_onehot(y) for y in y_true])
        y_pred = np.vstack(
            [y_pred[:, :2].sum(axis=1), y_pred[:, 2:].sum(axis=1)]
        ).transpose()

        d = y_pred[:, 1] - y_truel[:, 1]
        bs = np.dot(d, d) / N
        bsc = climatology[0] * climatology[1]
        bss = (bsc - bs) / bsc

        return bss

    def convert_binary_onehot(self, flare_class: int) -> List[int]:
        """
        return 2-dimentional 1-of-K vector
        """
        return np.array([1, 0] if flare_class < 2 else [0, 1])

    def confusion_matrix(self, y_pred: ndarray, y_true: ndarray) -> ndarray:
        """
        return confusion matrix for 4 class
        """
        return metrics.confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    def calc_accs(self, y_pred: ndarray, y_true: ndarray) -> float:
        """
        Compute classification accuracy for 4 class
        """
        cm = self.confusion_matrix(y_pred, y_true)
        accs = np.diag(cm).sum() / len(y_pred)
        print(cm)
        if self.logger is not None:
            self.logger.info(f"Confusion matrix: {cm}")

        return accs

    def binary_confusion_matrix(
        self, mtx: ndarray, target_class: int
    ) -> Tuple[int, int, int, int]:
        """
        convert confusion matrix for 2 class ("< target_class" or ">= target_class")
        """
        tn, fp = (
            mtx[:target_class, :target_class].sum(),
            mtx[:target_class, target_class:].sum(),
        )
        fn, tp = (
            mtx[target_class:, :target_class].sum(),
            mtx[target_class:, target_class:].sum(),
        )

        return tn, fp, fn, tp


def calculate_climatology(labels):
    """
    Function to calculate Climatology from class labels.
    Climatology represents the long-term average frequency of each class.
    """
    unique, counts = np.unique(labels, return_counts=True)
    climatology = counts / len(labels)
    return climatology


def calculate_gmgs_score_matrix(climatology):
    """
    Function to calculate GMGS Score Matrix from Climatology.
    """
    I = len(climatology)
    score_matrix = np.zeros((I, I))

    a = np.zeros(I)
    for i in range(I):
        a[i] = (1 - np.sum(climatology[: i + 1])) / np.sum(climatology[: i + 1])

    for i in range(I):
        for j in range(i, I):
            if i == j:
                score_matrix[i, j] = 1 / (I - 1) * (np.sum(a[:i]) + np.sum(a[i:]))
            else:
                score_matrix[i, j] = (
                    1
                    / (I - 1)
                    * (np.sum(a[:i]) + np.sum(-1 for k in range(i, j)) + np.sum(a[j:]))
                )

    # Score matrix is a symmetric matrix, so calculate the upper triangle and then transpose it
    for i in range(I):
        for j in range(i + 1, I):
            score_matrix[j, i] = score_matrix[i, j]

    return score_matrix


def read_labels_from_h5(file_path):
    """
    Read labels from H5 file and convert to numbers.
    """
    with h5py.File(file_path, "r") as f:
        try:
            if "y" not in f:
                return None

            dataset = f["y"]

            if dataset.shape == ():
                label = dataset[()]
            else:
                label = dataset[:]

            # Convert byte strings to numbers
            if isinstance(label, (bytes, np.bytes_)):
                label_str = label.decode("utf-8")
                # Convert labels to numbers
                label_map = {"O": 1, "C": 2, "M": 3, "X": 4}
                return np.array([label_map.get(label_str, 0)])
            elif isinstance(label, np.ndarray) and label.dtype.kind in ["S", "U"]:
                label_map = {b"O": 1, b"C": 2, b"M": 3, b"X": 4}
                return np.array([label_map.get(l, 0) for l in label])

            # If it's a single number, convert it to an array and return it
            if np.isscalar(label):
                return np.array([label])

            return label

        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return None


def find_h5_files(directory, start_date, end_date):
    """
    Find all H5 files in the directory that match the specified period.
    """
    files = [
        os.path.join(directory, fname)
        for fname in os.listdir(directory)
        if fname.endswith(".h5")
    ]
    filtered_files = []
    for file in files:
        date_str = os.path.basename(file).split(".")[0]  
        try:
            date = pd.to_datetime(date_str, format="%Y%m%d_%H%M%S")
            if start_date <= date <= end_date:
                filtered_files.append(file)
        except ValueError:
            continue
    return filtered_files


def compute_statistics(
    data_dir: str,
    stats_dir: str,
    train_periods: List[Tuple[str, str]],
    force_recalc: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray, Stat]:
    """Function to compute statistics"""
    if logger:
        logger.info("=" * 50)
        logger.info("Statistics Computation Start")
        logger.info(f"Using statistics directory: {stats_dir}")
        logger.info("Training periods:")
        for start, end in train_periods:
            logger.info(f"  {start} to {end}")
        logger.info("=" * 50)

    os.makedirs(stats_dir, exist_ok=True)
    climatology_file = os.path.join(stats_dir, "climatology.npy")
    gmgs_matrix_file = os.path.join(stats_dir, "gmgs_score_matrix.npy")

    if not force_recalc and os.path.exists(climatology_file):
        if logger:
            logger.info(f"Loading cached statistics from {stats_dir}")
        full_climatology = np.load(climatology_file)
        gmgs_score_matrix = np.load(gmgs_matrix_file)
        if logger:
            logger.info(f"Loaded full Climatology: {full_climatology}")
    else:
        all_labels = []
        for start_date, end_date in train_periods:
            train_h5_files = find_h5_files(
                data_dir, pd.to_datetime(start_date), pd.to_datetime(end_date)
            )
            if logger:
                logger.info(
                    f"Found {len(train_h5_files)} training files for period {start_date} to {end_date}"
                )

            for file in train_h5_files:
                labels = read_labels_from_h5(file)
                if labels is not None:
                    all_labels.extend(labels)

        all_labels = np.array(all_labels)
        if logger:
            logger.info(f"Total labels collected: {len(all_labels)}")
            logger.info(
                f"Label distribution: {np.unique(all_labels, return_counts=True)}"
            )

        filtered_labels = all_labels[(all_labels >= 1) & (all_labels <= 4)]
        if logger:
            logger.info(
                f"Filtered labels distribution: {np.unique(filtered_labels, return_counts=True)}"
            )

        full_climatology = calculate_climatology(filtered_labels)
        if logger:
            logger.info(f"Computed full Climatology: {full_climatology}")

        np.save(climatology_file, full_climatology)
        if logger:
            logger.info(f"Climatology saved to {climatology_file}")

        gmgs_score_matrix = calculate_gmgs_score_matrix(full_climatology)
        if logger:
            logger.info("Computed GMGS Score Matrix:")
            for i in range(len(gmgs_score_matrix)):
                logger.info(f"Row {i}: {gmgs_score_matrix[i]}")

        np.save(gmgs_matrix_file, gmgs_score_matrix)
        if logger:
            logger.info(f"GMGS Score Matrix saved to {gmgs_matrix_file}")

    climatology = np.array([full_climatology[0:2].sum(), full_climatology[2:4].sum()])
    climatology = climatology / climatology.sum()

    if logger:
        logger.info(f"Normalized Binary Climatology: {climatology}")
        logger.info(f"Below M class probability: {climatology[0]:.4f}")
        logger.info(f"M class and above probability: {climatology[1]:.4f}")
        logger.info("=" * 50)
        logger.info("Statistics Computation Complete")
        logger.info("=" * 50)

    stat = Stat(climatology, logger)
    stat.gmgs_score_matrix = gmgs_score_matrix

    return full_climatology, gmgs_score_matrix, stat
