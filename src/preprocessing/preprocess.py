# # File: src/preprocessing/preprocess.py
# import os
# import pandas as pd
# import re
# import numpy as np
# from scipy.signal import butter, filtfilt, resample
#
# # Keep only participants P01–P10
# PARTICIPANTS = {f"P{str(i).zfill(2)}" for i in range(1, 11)}
# # Map prefix to labels: '1'→0, '2'→2, '3'→1
# LABEL_MAP   = {'1': 0, '2': 2, '3': 1}
# # Regex to parse filenames like "1.P01_ECG"
# FNAME_RE    = re.compile(r"^([123])\.(P\d{2})_(ECG|EEG|Pupil)$")
#
# # Bandpass filter (ECG/EEG only)
# def bandpass_filter(sig_df, lowcut, highcut, fs, order=5):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     arr = filtfilt(b, a, sig_df.values, axis=0)
#     return pd.DataFrame(arr, columns=sig_df.columns)
#
# # Split into non-overlapping segments
# def segment_signal(df, fs, seg_sec):
#     seg_len = int(fs * seg_sec)
#     n_segs = len(df) // seg_len
#     return [
#         df.iloc[i*seg_len:(i+1)*seg_len].reset_index(drop=True)
#         for i in range(n_segs)
#     ]
#
# # Main preprocessing function
# def run_preprocessing(config, modality):
#     raw_dir   = os.path.join(config['data']['raw'], modality)
#     out_dir   = os.path.join(config['data']['processed'], modality)
#     os.makedirs(out_dir, exist_ok=True)
#
#     orig_fs   = 512 if modality in ('ECG', 'EEG') else 30
#     target_fs = config['preprocessing']['resample_rate']
#     seg_sec   = config['preprocessing']['segment_length_sec']
#     do_filter = orig_fs > 30
#     if do_filter:
#         lowcut, highcut = 0.5, 40.0
#
#     for fname in sorted(os.listdir(raw_dir)):
#         if not fname.lower().endswith('.csv'):
#             continue
#         base, _ = os.path.splitext(fname)
#         m = FNAME_RE.match(base)
#         if not m:
#             continue
#         prefix, pid, tag = m.groups()
#         if tag != modality or pid not in PARTICIPANTS:
#             continue
#
#         # Load raw
#         df_raw = pd.read_csv(os.path.join(raw_dir, fname))
#         # Drop original Time
#         sig_df = df_raw.drop(columns=['Time'])
#         # Filter if needed
#         if do_filter:
#             sig_df = bandpass_filter(sig_df, lowcut, highcut, orig_fs)
#         # Resample signals
#         n_new = int(len(sig_df) * target_fs / orig_fs)
#         arr   = resample(sig_df.values, n_new, axis=0)
#         df_rs = pd.DataFrame(arr, columns=sig_df.columns)
#         # Create uniform Time column
#         df_rs.insert(0, 'Time', np.arange(len(df_rs)) / target_fs)
#
#         # Segment and save
#         segments = segment_signal(df_rs, target_fs, seg_sec)
#         for idx, seg in enumerate(segments):
#             seg['label'] = LABEL_MAP[prefix]
#             out_name = f"{prefix}.{pid}_seg{idx}.csv"
#             seg.to_csv(os.path.join(out_dir, out_name), index=False)
