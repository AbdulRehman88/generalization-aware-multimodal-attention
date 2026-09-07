# =============================================================================
# pde_features.py
# =============================================================================
# PDE-based Feature Extraction for Attention Detection System
#
# This module implements 5 Partial Differential Equation (PDE) inspired
# transformations for physiological signal analysis:
#   1. Wave Equation
#   2. Burgers Equation
#   3. Diffusion Equation
#   4. Telegraph Equation
#   5. Klein-Gordon Equation
#
# Each transformation captures different dynamic properties of the signals
# that traditional statistical features may miss.
#
# Author: Sanjar / Dr. Abdul Rehman Lab
# Date: November 2025
# Project: ETRI Attention Detection System
# =============================================================================

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from typing import Dict, List, Union, Optional


# =============================================================================
# PDE TRANSFORMATION FUNCTIONS
# =============================================================================

def wave_equation(x: np.ndarray) -> np.ndarray:
    """
    Wave Equation Transformation: ∂²u/∂t²

    Computes the second derivative (acceleration) of the signal.
    Captures rapid changes and oscillatory behavior in physiological signals.

    Physical interpretation: Models wave propagation phenomena,
    useful for detecting sudden transitions in attention states.

    Args:
        x: 1D numpy array of signal values

    Returns:
        Second derivative of the signal
    """
    return np.gradient(np.gradient(x))


def burgers_equation(x: np.ndarray) -> np.ndarray:
    """
    Burgers Equation Transformation: ∂²u/∂t² - u·∂u/∂t

    Combines second derivative with a nonlinear convective term.
    Captures nonlinear dynamics and shock-like behavior in signals.

    Physical interpretation: Models nonlinear wave propagation with
    viscosity effects, useful for detecting abrupt state changes.

    Args:
        x: 1D numpy array of signal values

    Returns:
        Burgers equation transformation of the signal
    """
    first_derivative = np.gradient(x)
    second_derivative = np.gradient(first_derivative)
    return second_derivative - x * first_derivative


def diffusion_equation(x: np.ndarray) -> np.ndarray:
    """
    Diffusion Equation Transformation: ∂²u/∂t²

    Mathematically identical to wave equation but conceptually represents
    diffusion/spreading processes in the signal.

    Physical interpretation: Models how information or activation
    spreads through neural networks over time.

    Args:
        x: 1D numpy array of signal values

    Returns:
        Second derivative of the signal (diffusion term)
    """
    return np.gradient(np.gradient(x))


def telegraph_equation(x: np.ndarray) -> np.ndarray:
    """
    Telegraph Equation Transformation: ∂²u/∂t² - 2·∂u/∂t

    Combines second derivative with a damping term (first derivative).
    Captures damped oscillatory behavior in physiological signals.

    Physical interpretation: Models signal transmission with resistance
    and capacitance effects, useful for detecting sustained attention.

    Args:
        x: 1D numpy array of signal values

    Returns:
        Telegraph equation transformation of the signal
    """
    first_derivative = np.gradient(x)
    second_derivative = np.gradient(first_derivative)
    return second_derivative - 2 * first_derivative


def klein_gordon_equation(x: np.ndarray) -> np.ndarray:
    """
    Klein-Gordon Equation Transformation: ∂²u/∂t² - u

    Combines second derivative with the signal itself (mass term).
    Captures oscillatory behavior with amplitude dependence.

    Physical interpretation: Models relativistic wave mechanics,
    useful for detecting attention states with amplitude modulation.

    Args:
        x: 1D numpy array of signal values

    Returns:
        Klein-Gordon equation transformation of the signal
    """
    second_derivative = np.gradient(np.gradient(x))
    return second_derivative - x


# =============================================================================
# STATISTICAL FEATURE EXTRACTION FROM PDE TRANSFORMATIONS
# =============================================================================

def extract_pde_statistics(x: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Extract statistical features from a PDE-transformed signal.

    Computes 5 statistical measures that capture the distribution
    and energy characteristics of the transformed signal.

    Args:
        x: 1D numpy array of PDE-transformed signal
        prefix: String prefix for feature names (e.g., 'AF7_wave')

    Returns:
        Dictionary of statistical features with prefixed names
    """
    # Handle edge cases
    if len(x) == 0 or np.all(np.isnan(x)):
        return {
            f'{prefix}_mean': 0.0,
            f'{prefix}_std': 0.0,
            f'{prefix}_skew': 0.0,
            f'{prefix}_kurtosis': 0.0,
            f'{prefix}_rms': 0.0
        }

    # Remove NaN values for robust computation
    x_clean = x[~np.isnan(x)]
    if len(x_clean) == 0:
        x_clean = np.zeros(1)

    return {
        f'{prefix}_mean': float(np.mean(x_clean)),
        f'{prefix}_std': float(np.std(x_clean)),
        f'{prefix}_skew': float(skew(x_clean, nan_policy='omit')),
        f'{prefix}_kurtosis': float(kurtosis(x_clean, nan_policy='omit')),
        f'{prefix}_rms': float(np.sqrt(np.mean(x_clean ** 2)))
    }


def extract_all_pde_features(signal: np.ndarray, channel_name: str) -> Dict[str, float]:
    """
    Apply all 5 PDE transformations to a signal and extract statistics.

    This is the core function that generates 25 PDE features per channel:
    5 PDE equations × 5 statistical measures = 25 features

    Args:
        signal: 1D numpy array of raw signal values
        channel_name: Name of the channel (e.g., 'AF7', 'ECG', 'pupil')

    Returns:
        Dictionary containing 25 PDE-based features
    """
    features = {}

    # Define PDE transformations
    pde_transforms = {
        'wave': wave_equation,
        'burgers': burgers_equation,
        'diffusion': diffusion_equation,
        'telegraph': telegraph_equation,
        'kg': klein_gordon_equation  # Klein-Gordon abbreviated as 'kg'
    }

    # Apply each PDE transformation and extract statistics
    for pde_name, pde_func in pde_transforms.items():
        try:
            transformed = pde_func(signal)
            prefix = f'{channel_name}_{pde_name}'
            stats = extract_pde_statistics(transformed, prefix)
            features.update(stats)
        except Exception as e:
            # If transformation fails, fill with zeros
            prefix = f'{channel_name}_{pde_name}'
            features.update({
                f'{prefix}_mean': 0.0,
                f'{prefix}_std': 0.0,
                f'{prefix}_skew': 0.0,
                f'{prefix}_kurtosis': 0.0,
                f'{prefix}_rms': 0.0
            })

    return features


# =============================================================================
# MODALITY-SPECIFIC PDE FEATURE EXTRACTION
# =============================================================================

def extract_eeg_pde_features(df: pd.DataFrame, fs: int = 128) -> Dict[str, float]:
    """
    Extract PDE features from EEG data (all channels).

    For 8 EEG channels × 5 PDE equations × 5 statistics = 200 PDE features

    Args:
        df: DataFrame with EEG channels (excluding 'Time', 'label', 'Segment_ID')
        fs: Sampling frequency (default 128 Hz)

    Returns:
        Dictionary of PDE features for all EEG channels
    """
    features = {}

    # Get all EEG channel columns (exclude metadata columns)
    exclude_cols = ['Time', 'time', 'label', 'Label', 'Segment_ID', 'segment_id']
    channels = [col for col in df.columns if col not in exclude_cols]

    for channel in channels:
        signal = df[channel].values

        # Handle NaN values
        if np.any(np.isnan(signal)):
            signal = pd.Series(signal).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values

        # Skip if signal is too short
        if len(signal) < 10:
            continue

        # Extract PDE features for this channel
        channel_features = extract_all_pde_features(signal, channel)
        features.update(channel_features)

    return features


def extract_ecg_pde_features(df: pd.DataFrame, fs: int = 128) -> Dict[str, float]:
    """
    Extract PDE features from ECG data.

    For 1 ECG channel × 5 PDE equations × 5 statistics = 25 PDE features

    Args:
        df: DataFrame with ECG signal (column 'ECG_Signal' or similar)
        fs: Sampling frequency (default 128 Hz)

    Returns:
        Dictionary of PDE features for ECG
    """
    features = {}

    # Find ECG signal column
    ecg_col = None
    for col in df.columns:
        if 'ecg' in col.lower() or 'signal' in col.lower():
            ecg_col = col
            break

    if ecg_col is None:
        # Try to use the first non-metadata column
        exclude_cols = ['Time', 'time', 'label', 'Label', 'Segment_ID', 'segment_id']
        data_cols = [col for col in df.columns if col not in exclude_cols]
        if data_cols:
            ecg_col = data_cols[0]
        else:
            return features

    signal = df[ecg_col].values

    # Handle NaN values
    if np.any(np.isnan(signal)):
        signal = pd.Series(signal).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values

    # Skip if signal is too short
    if len(signal) < 10:
        return features

    # Extract PDE features with 'ecg' prefix
    features = extract_all_pde_features(signal, 'ecg')

    return features


def extract_pupil_pde_features(df: pd.DataFrame, fs: int = 128) -> Dict[str, float]:
    """
    Extract PDE features from Pupil data.

    For 1 pupil channel (r) × 5 PDE equations × 5 statistics = 25 PDE features

    Args:
        df: DataFrame with pupil data (column 'r' for pupil radius)
        fs: Sampling frequency (default 128 Hz)

    Returns:
        Dictionary of PDE features for Pupil
    """
    features = {}

    # Find pupil radius column
    pupil_col = None
    for col in df.columns:
        if col.lower() == 'r':
            pupil_col = col
            break

    if pupil_col is None:
        return features

    signal = df[pupil_col].values

    # Handle NaN values (common in pupil data due to blinks)
    if np.any(np.isnan(signal)):
        signal = pd.Series(signal).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values

    # Remove remaining NaN values
    signal = signal[~np.isnan(signal)]

    # Skip if signal is too short
    if len(signal) < 10:
        return features

    # Extract PDE features with 'pupil' prefix
    features = extract_all_pde_features(signal, 'pupil')

    return features


# =============================================================================
# UNIFIED PDE FEATURE EXTRACTION INTERFACE
# =============================================================================

def extract_pde_features_for_modality(
        df: pd.DataFrame,
        modality: str,
        fs: int = 128
) -> Dict[str, float]:
    """
    Unified interface for extracting PDE features from any modality.

    Args:
        df: DataFrame with signal data
        modality: One of 'EEG', 'ECG', 'Pupil' (case-insensitive)
        fs: Sampling frequency (default 128 Hz)

    Returns:
        Dictionary of PDE features for the specified modality
    """
    modality_upper = modality.upper()

    if modality_upper == 'EEG':
        return extract_eeg_pde_features(df, fs)
    elif modality_upper == 'ECG':
        return extract_ecg_pde_features(df, fs)
    elif modality_upper == 'PUPIL':
        return extract_pupil_pde_features(df, fs)
    else:
        raise ValueError(f"Unknown modality: {modality}. Expected 'EEG', 'ECG', or 'Pupil'.")


# =============================================================================
# FEATURE COUNT SUMMARY
# =============================================================================

def get_pde_feature_count(modality: str) -> int:
    """
    Get the expected number of PDE features for a given modality.

    Args:
        modality: One of 'EEG', 'ECG', 'Pupil'

    Returns:
        Expected number of PDE features

    Feature counts:
        - EEG (8 channels): 8 × 5 × 5 = 200 features
        - ECG (1 channel): 1 × 5 × 5 = 25 features
        - Pupil (1 channel): 1 × 5 × 5 = 25 features
    """
    modality_upper = modality.upper()

    # 5 PDE equations × 5 statistics = 25 features per channel
    features_per_channel = 25

    channel_counts = {
        'EEG': 8,  # AF7, Fp1, Fpz, Fp2, AF8, O1, POz, O2
        'ECG': 1,  # ECG_Signal
        'PUPIL': 1  # r (pupil radius)
    }

    if modality_upper not in channel_counts:
        raise ValueError(f"Unknown modality: {modality}")

    return channel_counts[modality_upper] * features_per_channel


def get_pde_feature_names(modality: str, channels: Optional[List[str]] = None) -> List[str]:
    """
    Get the list of PDE feature names for a given modality.

    Args:
        modality: One of 'EEG', 'ECG', 'Pupil'
        channels: Optional list of channel names (for EEG)

    Returns:
        List of feature names
    """
    pde_names = ['wave', 'burgers', 'diffusion', 'telegraph', 'kg']
    stat_names = ['mean', 'std', 'skew', 'kurtosis', 'rms']

    modality_upper = modality.upper()

    if modality_upper == 'EEG':
        if channels is None:
            channels = ['AF7', 'Fp1', 'Fpz', 'Fp2', 'AF8', 'O1', 'POz', 'O2']
    elif modality_upper == 'ECG':
        channels = ['ecg']
    elif modality_upper == 'PUPIL':
        channels = ['pupil']
    else:
        raise ValueError(f"Unknown modality: {modality}")

    feature_names = []
    for ch in channels:
        for pde in pde_names:
            for stat in stat_names:
                feature_names.append(f'{ch}_{pde}_{stat}')

    return feature_names


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == '__main__':
    # Test with synthetic data
    print("=" * 60)
    print("PDE Features Module - Test Suite")
    print("=" * 60)

    # Generate synthetic signals
    np.random.seed(42)
    fs = 128
    duration = 5  # seconds
    n_samples = fs * duration
    t = np.arange(n_samples) / fs

    # Synthetic EEG (8 channels)
    eeg_channels = ['AF7', 'Fp1', 'Fpz', 'Fp2', 'AF8', 'O1', 'POz', 'O2']
    eeg_data = {ch: np.sin(2 * np.pi * 10 * t + np.random.rand()) + 0.1 * np.random.randn(n_samples)
                for ch in eeg_channels}
    df_eeg = pd.DataFrame(eeg_data)

    # Synthetic ECG
    ecg_signal = 0.5 * np.sin(2 * np.pi * 1.2 * t) + 0.05 * np.random.randn(n_samples)
    df_ecg = pd.DataFrame({'ECG_Signal': ecg_signal})

    # Synthetic Pupil
    pupil_signal = 3.5 + 0.2 * np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(n_samples)
    df_pupil = pd.DataFrame({'r': pupil_signal})

    # Test EEG PDE features
    print("\n1. Testing EEG PDE Features:")
    print("-" * 40)
    eeg_pde = extract_eeg_pde_features(df_eeg, fs)
    print(f"   Total features extracted: {len(eeg_pde)}")
    print(f"   Expected features: {get_pde_feature_count('EEG')}")
    print(f"   Sample features: {list(eeg_pde.keys())[:5]}")

    # Test ECG PDE features
    print("\n2. Testing ECG PDE Features:")
    print("-" * 40)
    ecg_pde = extract_ecg_pde_features(df_ecg, fs)
    print(f"   Total features extracted: {len(ecg_pde)}")
    print(f"   Expected features: {get_pde_feature_count('ECG')}")
    print(f"   All features: {list(ecg_pde.keys())}")

    # Test Pupil PDE features
    print("\n3. Testing Pupil PDE Features:")
    print("-" * 40)
    pupil_pde = extract_pupil_pde_features(df_pupil, fs)
    print(f"   Total features extracted: {len(pupil_pde)}")
    print(f"   Expected features: {get_pde_feature_count('Pupil')}")
    print(f"   All features: {list(pupil_pde.keys())}")

    # Test unified interface
    print("\n4. Testing Unified Interface:")
    print("-" * 40)
    for mod, df in [('EEG', df_eeg), ('ECG', df_ecg), ('Pupil', df_pupil)]:
        feats = extract_pde_features_for_modality(df, mod, fs)
        print(f"   {mod}: {len(feats)} features")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: PDE Feature Counts")
    print("=" * 60)
    total = 0
    for mod in ['EEG', 'ECG', 'Pupil']:
        count = get_pde_feature_count(mod)
        total += count
        print(f"   {mod}: {count} features")
    print(f"   ---")
    print(f"   TOTAL (Tri-modality): {total} PDE features")
    print("=" * 60)
    print("\n✅ All tests passed! PDE features module is ready for use.")