# =============================================================================
# extract_features.py
# =============================================================================
# Feature Extraction Module for Attention Detection System
# 
# This module extracts BOTH Traditional and PDE-based features from
# physiological signals (EEG, ECG, Pupil).
#
# Features extracted:
#   - Traditional: Time-domain, Frequency-domain, Band powers, Ratios
#   - PDE-based: Wave, Burgers, Diffusion, Telegraph, Klein-Gordon
#
# Author: Sanjar / Dr. Abdul Rehman Lab
# Date: November 2025
# Project: ETRI Attention Detection System
#
# IMPORTANT: All original naming conventions are preserved for compatibility
# =============================================================================

import os
import yaml
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis
from pathlib import Path


# =============================================================================
# PDE FEATURE FUNCTIONS (Integrated from pde_features.py)
# =============================================================================

def wave_equation(x):
    """Wave Equation: ∂²u/∂t² - Second derivative of signal"""
    return np.gradient(np.gradient(x))


def burgers_equation(x):
    """Burgers Equation: ∂²u/∂t² - u·∂u/∂t - Nonlinear dynamics"""
    g = np.gradient(x)
    return np.gradient(g) - x * g


def diffusion_equation(x):
    """Diffusion Equation: ∂²u/∂t² - Information spreading"""
    return np.gradient(np.gradient(x))


def telegraph_equation(x):
    """Telegraph Equation: ∂²u/∂t² - 2·∂u/∂t - Damped oscillations"""
    return np.gradient(np.gradient(x)) - 2 * np.gradient(x)


def klein_gordon_equation(x):
    """Klein-Gordon Equation: ∂²u/∂t² - u - Amplitude-dependent oscillations"""
    return np.gradient(np.gradient(x)) - x


def extract_pde_statistics(x, prefix):
    """Extract 5 statistical features from PDE-transformed signal"""
    if len(x) == 0 or np.all(np.isnan(x)):
        return {
            f'{prefix}_mean': 0.0,
            f'{prefix}_std': 0.0,
            f'{prefix}_skew': 0.0,
            f'{prefix}_kurtosis': 0.0,
            f'{prefix}_rms': 0.0
        }

    x_clean = x[~np.isnan(x)]
    if len(x_clean) == 0:
        x_clean = np.zeros(1)

    return {
        f'{prefix}_mean': float(np.mean(x_clean)),
        f'{prefix}_std': float(np.std(x_clean)),
        f'{prefix}_skew': float(scipy_skew(x_clean, nan_policy='omit')),
        f'{prefix}_kurtosis': float(scipy_kurtosis(x_clean, nan_policy='omit')),
        f'{prefix}_rms': float(np.sqrt(np.mean(x_clean ** 2)))
    }


def extract_channel_pde_features(signal, channel_name):
    """
    Apply all 5 PDE transformations and extract statistics.
    Returns 25 features per channel (5 PDEs × 5 stats)
    """
    features = {}

    pde_transforms = {
        'wave': wave_equation,
        'burgers': burgers_equation,
        'diffusion': diffusion_equation,
        'telegraph': telegraph_equation,
        'kg': klein_gordon_equation
    }

    for pde_name, pde_func in pde_transforms.items():
        try:
            transformed = pde_func(signal)
            prefix = f'{channel_name}_{pde_name}'
            stats = extract_pde_statistics(transformed, prefix)
            features.update(stats)
        except Exception:
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
# TRADITIONAL FEATURE FUNCTIONS (Original - Unchanged)
# =============================================================================

def spectral_entropy(x, fs):
    """Compute spectral entropy of signal"""
    freqs, psd = welch(x, fs=fs, nperseg=min(len(x), 256))
    psd_norm = psd / (np.sum(psd) + 1e-12)
    psd_norm = np.clip(psd_norm, 1e-12, None)
    return -np.sum(psd_norm * np.log(psd_norm))


def band_power(x, fs, band):
    """Compute band power in specified frequency range"""
    freqs, psd = welch(x, fs=fs, nperseg=min(len(x), 256))
    idx = (freqs >= band[0]) & (freqs < band[1])
    if np.sum(idx) == 0:
        return 0.0
    # Use numpy's trapezoid (newer) or trapz (legacy) for compatibility
    try:
        return np.trapezoid(psd[idx], freqs[idx])
    except AttributeError:
        return np.trapz(psd[idx], freqs[idx])


def extract_time_features(x, fs):
    """Extract time-domain features from signal (Original naming preserved)"""
    arr = np.asarray(x)
    feats = {
        'mean': np.mean(arr),
        'std': np.std(arr),
        'min': np.min(arr),
        'max': np.max(arr),
        'range': np.max(arr) - np.min(arr),
        'rms': np.sqrt(np.mean(arr ** 2)),
        'skew': pd.Series(arr).skew(),
        'kurtosis': pd.Series(arr).kurtosis(),
        'sma': np.sum(np.abs(arr)) / len(arr),
        'spec_entropy': spectral_entropy(arr, fs)
    }
    return feats


# =============================================================================
# MODALITY-SPECIFIC EXTRACTION (Original naming preserved + PDE added)
# =============================================================================

def extract_eeg_features(df, fs):
    """
    Extract EEG features: Traditional + PDE

    Traditional: ~23 features per channel (time-domain + band powers + ratios)
    PDE: 25 features per channel (5 PDEs × 5 stats)
    Total: ~48 features per channel × 8 channels = ~384 features

    Original naming convention preserved exactly.
    """
    # Dynamically grab all columns except time & label
    channels = [c for c in df.columns if c not in ("Time", "time", "label", "Label", "Segment_ID", "segment_id")]

    bands = {
        'delta': (0.5, 4), 'theta': (4, 8),
        'low_alpha': (8, 10), 'high_alpha': (10, 12),
        'low_beta': (12, 18), 'high_beta': (18, 30),
        'low_gamma': (30, 45), 'high_gamma': (45, 64)
    }

    feats = {}

    for ch in channels:
        x = df[ch].values

        # Handle NaN values
        if np.any(np.isnan(x)):
            x = pd.Series(x).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values

        # Skip if signal too short
        if len(x) < 10:
            continue

        # =====================
        # TRADITIONAL FEATURES (Original naming preserved)
        # =====================

        # Time-domain features
        tfeat = extract_time_features(x, fs)
        for k, v in tfeat.items():
            feats[f"{ch}_{k}"] = v

        # Band powers
        bp = {}
        for name, band in bands.items():
            bp[name] = band_power(x, fs, band)
            feats[f"{ch}_bp_{name}"] = bp[name]

        # Ratios (Original naming preserved exactly)
        theta = bp['theta']
        alpha = bp['low_alpha'] + bp['high_alpha']
        beta = bp['low_beta'] + bp['high_beta']
        feats[f"{ch}_ratio_theta_alpha"] = theta / (alpha + 1e-12)
        feats[f"{ch}_ratio_theta_beta"] = theta / (beta + 1e-12)
        feats[f"{ch}_ratio_alpha_beta"] = alpha / (beta + 1e-12)
        feats[f"{ch}_ratio_beta_theta"] = beta / (theta + 1e-12)
        feats[f"{ch}_ratio_beta_(a+t)"] = beta / (alpha + theta + 1e-12)

        # =====================
        # PDE FEATURES (New addition)
        # =====================
        pde_feats = extract_channel_pde_features(x, ch)
        feats.update(pde_feats)

    return feats


def extract_ecg_features(df, fs):
    """
    Extract ECG features: Traditional + PDE

    Traditional: ~14 features (time-domain + HRV bands + ratio)
    PDE: 25 features (5 PDEs × 5 stats)
    Total: ~39 features

    Original naming convention preserved exactly.
    """
    # Find ECG signal column
    ecg_col = 'ECG_Signal'
    if ecg_col not in df.columns:
        # Try to find it
        for col in df.columns:
            if 'ecg' in col.lower() or 'signal' in col.lower():
                ecg_col = col
                break

    x = df[ecg_col].values

    # Handle NaN values
    if np.any(np.isnan(x)):
        x = pd.Series(x).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values

    # =====================
    # TRADITIONAL FEATURES (Original naming preserved)
    # =====================
    feats = extract_time_features(x, fs)

    # HRV frequency bands
    hrv_bands = {'vlf': (0.003, 0.04), 'lf': (0.04, 0.15), 'hf': (0.15, 0.4)}
    for name, band in hrv_bands.items():
        feats[f"ecg_{name}_power"] = band_power(x, fs, band)

    # LF/HF ratio (Original naming preserved)
    feats['ecg_ratio_lf_hf'] = feats['ecg_lf_power'] / (feats['ecg_hf_power'] + 1e-12)

    # =====================
    # PDE FEATURES (New addition)
    # =====================
    pde_feats = extract_channel_pde_features(x, 'ecg')
    feats.update(pde_feats)

    return feats


def extract_pupil_features(df, fs):
    """
    Extract Pupil features: Traditional + PDE

    Traditional: ~17 features (pupil size, velocity, gaze)
    PDE: 25 features (5 PDEs × 5 stats)
    Total: ~42 features

    Original naming convention preserved exactly.
    """
    feats = {}

    # =====================
    # TRADITIONAL FEATURES (Original naming preserved)
    # =====================

    # Pupil dilation (r column)
    x = df['r'].values

    # Handle NaN values (common in pupil data due to blinks)
    if np.any(np.isnan(x)):
        x = pd.Series(x).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values

    # Time-domain features for pupil
    pfeat = extract_time_features(x, fs)
    for k, v in pfeat.items():
        feats[f"pupil_{k}"] = v

    # Velocity features
    vel = np.diff(x) * fs
    feats['pupil_vel_mean'] = np.mean(vel)
    feats['pupil_vel_max'] = np.max(vel)
    feats['pupil_vel_std'] = np.std(vel)

    # Frequency energy
    feats['pupil_energy_lf'] = band_power(x, fs, (0, 4))
    feats['pupil_energy_hf'] = band_power(x, fs, (4, fs / 2))

    # Gaze position features (Original naming preserved)
    for axis in ('x', 'y'):
        if axis in df.columns:
            yv = df[axis].values
            if np.any(np.isnan(yv)):
                yv = pd.Series(yv).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
            feats[f"{axis}_mean"] = np.mean(yv)
            feats[f"{axis}_std"] = np.std(yv)

    # =====================
    # PDE FEATURES (New addition)
    # =====================
    x_clean = x[~np.isnan(x)] if np.any(np.isnan(x)) else x
    if len(x_clean) >= 10:
        pde_feats = extract_channel_pde_features(x_clean, 'pupil')
        feats.update(pde_feats)

    return feats


# =============================================================================
# MAIN EXTRACTION & FUSION (Original logic preserved)
# =============================================================================

def main():
    """
    Main feature extraction pipeline.

    1. Extracts Traditional + PDE features for each single modality
    2. Fuses features for dual/triple modality combinations
    3. Saves all feature files to the configured output directory

    Output files (same naming as original):
        - ECG_features.csv
        - EEG_features.csv
        - Pupil_features.csv
        - ECG_EEG_features.csv
        - ECG_Pupil_features.csv
        - EEG_Pupil_features.csv
        - ECG_EEG_Pupil_features.csv
    """
    # Locate config.yaml (two levels up from this file)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    cfg_path = PROJECT_ROOT / 'configs' / 'config.yaml'

    if not cfg_path.exists():
        # Fallback: try current directory structure
        cfg_path = Path('configs/config.yaml')

    # Load config
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    fs = cfg['preprocessing']['resample_rate']
    in_base = cfg['data']['processed']
    out_dir = cfg['features']['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("FEATURE EXTRACTION: Traditional + PDE Features")
    print("=" * 60)

    # =========================
    # SINGLE MODALITIES
    # =========================
    for mod in ['ECG', 'EEG', 'Pupil']:
        print(f"\n[{mod}] Processing...")
        rows = []
        path = os.path.join(in_base, mod)

        if not os.path.exists(path):
            print(f"  ⚠️ Warning: Path not found: {path}")
            continue

        files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
        total_files = len(files)

        for idx, fname in enumerate(files):
            if (idx + 1) % 500 == 0 or idx == 0:
                print(f"  Processing file {idx + 1}/{total_files}...")

            try:
                df = pd.read_csv(os.path.join(path, fname))
                label = df['label'].iloc[0]
                data = df.drop(columns=['label'], errors='ignore')

                if mod == 'EEG':
                    feats = extract_eeg_features(data, fs)
                elif mod == 'ECG':
                    feats = extract_ecg_features(data, fs)
                else:
                    feats = extract_pupil_features(data, fs)

                feats['segment_id'] = fname
                feats['label'] = label
                rows.append(feats)

            except Exception as e:
                print(f"  ⚠️ Error processing {fname}: {e}")
                continue

        if rows:
            df_out = pd.DataFrame(rows)
            out_path = os.path.join(out_dir, f"{mod}_features.csv")
            df_out.to_csv(out_path, index=False)
            n_features = len(df_out.columns) - 2  # Exclude segment_id and label
            print(f"  ✅ Saved {mod} features → {out_path}")
            print(f"     Samples: {len(df_out)} | Features: {n_features}")

    # =========================
    # FUSION FOR DUAL/TRIPLE
    # =========================
    print("\n" + "-" * 60)
    print("FUSING MODALITIES")
    print("-" * 60)

    modalities = cfg['modalities']

    for mod in modalities:
        parts = mod.split('_')
        if len(parts) == 1:
            continue  # Skip single modalities (already processed)

        print(f"\n[{mod}] Fusing {' + '.join(parts)}...")

        try:
            # Load feature files for each part
            dfs = []
            for p in parts:
                feat_path = os.path.join(out_dir, f"{p}_features.csv")
                if not os.path.exists(feat_path):
                    raise FileNotFoundError(f"Missing: {feat_path}")
                dfs.append(pd.read_csv(feat_path))

            # Merge on segment_id
            df_fused = dfs[0]
            for df2 in dfs[1:]:
                df_fused = df_fused.merge(
                    df2.drop(columns=['label'], errors='ignore'),
                    on='segment_id',
                    how='inner'
                )

            out_path = os.path.join(out_dir, f"{mod}_features.csv")
            df_fused.to_csv(out_path, index=False)
            n_features = len(df_fused.columns) - 2  # Exclude segment_id and label
            print(f"  ✅ Saved fused {mod} → {out_path}")
            print(f"     Samples: {len(df_fused)} | Features: {n_features}")

        except Exception as e:
            print(f"  ⚠️ Error fusing {mod}: {e}")
            continue

    # =========================
    # SUMMARY
    # =========================
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)

    # List all generated files
    print("\nGenerated feature files:")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith('_features.csv'):
            fpath = os.path.join(out_dir, f)
            df_check = pd.read_csv(fpath)
            n_feat = len(df_check.columns) - 2
            print(f"  • {f}: {len(df_check)} samples × {n_feat} features")


# =============================================================================
# UTILITY FUNCTIONS FOR EXTERNAL USE
# =============================================================================

def get_feature_count_summary():
    """
    Returns expected feature counts per modality.

    Traditional features:
        - EEG: ~23 per channel × 8 channels = ~184
        - ECG: ~14
        - Pupil: ~17

    PDE features:
        - EEG: 25 per channel × 8 channels = 200
        - ECG: 25
        - Pupil: 25

    Combined (Traditional + PDE):
        - EEG: ~384 features
        - ECG: ~39 features
        - Pupil: ~42 features
        - ECG_EEG: ~423 features
        - ECG_Pupil: ~81 features
        - EEG_Pupil: ~426 features
        - ECG_EEG_Pupil: ~465 features
    """
    summary = {
        'EEG': {'traditional': 184, 'pde': 200, 'total': 384},
        'ECG': {'traditional': 14, 'pde': 25, 'total': 39},
        'Pupil': {'traditional': 17, 'pde': 25, 'total': 42},
        'ECG_EEG': {'total': 423},
        'ECG_Pupil': {'total': 81},
        'EEG_Pupil': {'total': 426},
        'ECG_EEG_Pupil': {'total': 465}
    }
    return summary


if __name__ == '__main__':
    main()