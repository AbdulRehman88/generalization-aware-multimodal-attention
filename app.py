# # app.py
#
# import os
# from pathlib import Path
#
# import streamlit as st
# import yaml
# import pandas as pd
# import joblib
# import shap
# import numpy as np
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
# import mne
# from mne.viz import plot_topomap
#
# # your on‑the‑fly feature extractors
# from src.features.extract_features import (
#     extract_eeg_features,
#     extract_ecg_features,
#     extract_pupil_features,
# )
#
# st.set_page_config(layout="wide")
# st.title("🔴 Real‑Time Multimodal Attention Detection")
#
# # ─── Sidebar: config paths ──────────────────────────────────────────────────────
# st.sidebar.header("Configuration")
#
# config_path = st.sidebar.text_input(
#     "Path to config.yaml",
#     value=str(Path(__file__).resolve().parent / "configs" / "config.legacy.yaml")
# )
# if not os.path.isfile(config_path):
#     st.sidebar.error("❌ Invalid config.yaml path")
#     st.stop()
#
# # load YAML
# with open(config_path, "r", encoding="utf-8") as f:
#     config = yaml.safe_load(f)
#
# # auto‑locate the BioSemi coords CSV
# proj_root = Path(config["project"]["root"])
# coord_csv = proj_root / "configs" / "biosemi_64 ch_cap_coords.csv"
# if not coord_csv.exists():
#     st.sidebar.error(f"❌ Coord CSV not found at {coord_csv}")
#     st.stop()
#
# # base dirs & modality list
# processed_dir = Path(config["data"]["processed"])
# model_dir     = Path(config["models"]["output_dir"])
# sel_txt_dir   = processed_dir / "selected_features"
# modalities    = config["modalities"]
#
# # ─── Sidebar: pick modalities & segment files ─────────────────────────────────
# st.sidebar.header("Select Modalities & Segment Files")
#
# selected_modalities = st.sidebar.multiselect(
#     "Modalities",
#     modalities,
#     default=["ECG","EEG","Pupil"]
# )
# if not selected_modalities:
#     st.sidebar.error("❌ Pick at least one modality")
#     st.stop()
#
# mod_paths = {}
# for mod in selected_modalities:
#     folder = processed_dir / mod
#     if not folder.exists():
#         st.sidebar.error(f"Folder not found: {folder}")
#         st.stop()
#     files = sorted([f.name for f in folder.glob("*.csv")])
#     sel   = st.sidebar.selectbox(f"{mod} segment", files, key=mod)
#     mod_paths[mod] = folder / sel
#
# # ─── Run Detection ─────────────────────────────────────────────────────────────
# if st.sidebar.button("▶️ Run Detection"):
#     try:
#         # 1) load model
#         combo_key = "_".join(selected_modalities)
#         clf       = "lgbm"  # your best classifier
#         model_fp  = model_dir / f"{combo_key}_{clf}.pkl"
#         if not model_fp.exists():
#             raise FileNotFoundError(f"Model not found: {model_fp}")
#         model = joblib.load(model_fp)
#
#         # 2) extract features on‑the‑fly
#         fs = config["preprocessing"]["resample_rate"]
#         feat_dict = {}
#         for mod, fp in mod_paths.items():
#             df_raw = pd.read_csv(fp)
#             data_df = df_raw.drop(columns=["Time","label"], errors="ignore")
#             if mod == "EEG":
#                 feats = extract_eeg_features(data_df, fs)
#             elif mod == "ECG":
#                 feats = extract_ecg_features(data_df, fs)
#             elif mod == "Pupil":
#                 feats = extract_pupil_features(data_df, fs)
#             else:
#                 raise ValueError(f"Unknown modality: {mod}")
#             feat_dict.update(feats)
#         X_all = pd.DataFrame([feat_dict])
#
#         # 3) load SHAP‑selected feature names
#         feat_txt = sel_txt_dir / f"{combo_key}_top20_features.txt"
#         if not feat_txt.exists():
#             raise FileNotFoundError(f"Feature list not found: {feat_txt}")
#         with open(feat_txt, "r", encoding="utf-8") as f:
#             feature_list = [ln.strip() for ln in f if ln.strip()]
#         X_feat = X_all[feature_list]
#
#         # 4) predict
#         preds = model.predict(X_feat)
#         label_map = {0:"Low",1:"Mid",2:"High"}
#         maj_lab = label_map[np.bincount(preds).argmax()]
#
#         # ── Display: Attention Gauge ───────────────────────────
#         st.subheader("Attention Level")
#         gauge = go.Figure(
#             go.Indicator(
#                 mode="gauge+number",
#                 value=["Low","Mid","High"].index(maj_lab),
#                 number={"suffix":f"  ({maj_lab})"},
#                 gauge={"axis":{"range":[0,2],"tickvals":[0,1,2],"ticktext":["Low","Mid","High"]}}
#             )
#         )
#         gauge.update_layout(height=300)
#         st.plotly_chart(gauge, use_container_width=True)
#
#         # ── Display: SHAP Beeswarm ─────────────────────────────
#         st.subheader("SHAP Beeswarm (Top‑5 Features)")
#         explainer = shap.TreeExplainer(model)
#         shap_vals = explainer.shap_values(X_feat)
#         plt.figure(figsize=(6,4))
#         shap.summary_plot(shap_vals, X_feat, plot_type="dot", max_display=5, show=False)
#         st.pyplot(plt.gcf())
#
#         # ── Display: EEG Topomap ───────────────────────────────
#         if "EEG" in selected_modalities:
#             st.subheader("EEG Scalp Topography")
#
#             # raw EEG timeseries
#             df_eeg = pd.read_csv(mod_paths["EEG"])
#
#             # channel‑wise metric: mean absolute voltage
#             ch_vals = df_eeg.abs().mean(axis=0)
#
#             # read coords
#             coord_df = pd.read_csv(coord_csv, header=33, encoding="latin-1")
#             cols     = coord_df.columns
#             elec_col = [c for c in cols if "Electrode" in c][0]
#             x_col    = [c for c in cols if c.strip().startswith("x")][0]
#             y_col    = [c for c in cols if c.strip().startswith("y")][0]
#             coord_df = coord_df[[elec_col, x_col, y_col]].dropna()
#             coords   = {
#                 row[elec_col].strip(): (float(row[x_col]), float(row[y_col]))
#                 for _, row in coord_df.iterrows()
#             }
#
#             # build data & pos arrays
#             ch_present = [c for c in ch_vals.index if c in coords]
#             data  = np.array([ch_vals[c] for c in ch_present])
#             pos   = np.array([coords[c] for c in ch_present])
#
#             # plot
#             fig, ax = plt.subplots()
#             plot_topomap(data, pos, axes=ax, show=False)
#             ax.set_title("Mean |Voltage| per Channel")
#             st.pyplot(fig)
#
#         else:
#             st.warning("EEG not selected → no topography")
#
#     except Exception as e:
#         st.error(f"❌ Error during detection:\n{e}")
#
#
#
#
#
#

# #============================================================================================== Code 2 ===============
#
#
#
# # app.py
#
# import os
# import re
# from pathlib import Path
#
# import streamlit as st
# import yaml
# import pandas as pd
# import joblib
# import shap
# import numpy as np
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
# import mne
# from mne.viz import plot_topomap
#
# from src.features.extract_features import (
#     extract_eeg_features,
#     extract_ecg_features,
#     extract_pupil_features,
# )
#
# st.set_page_config(layout="wide")
# st.title("🔴 Real‑Time Multimodal Attention Detection")
#
# # ─── Sidebar: Config ────────────────────────────────────────────────────────────
# st.sidebar.header("Configuration")
#
# config_path = st.sidebar.text_input(
#     "Path to config.yaml",
#     value=str(Path(__file__).resolve().parent / "configs" / "config.legacy.yaml")
# )
# if not os.path.isfile(config_path):
#     st.sidebar.error("❌ Invalid config.yaml path")
#     st.stop()
#
# with open(config_path, "r", encoding="utf-8") as f:
#     config = yaml.safe_load(f)
#
# proj_root     = Path(config["project"]["root"])
# configs_dir   = proj_root / "configs"
# processed_dir = Path(config["data"]["processed"])
# model_dir     = Path(config["models"]["output_dir"])
# sel_txt_dir   = processed_dir / "selected_features"
# modalities    = config["modalities"]
#
# # find your BioSemi coords CSV automatically
# csvs     = list(configs_dir.glob("*.csv"))
# biosemi  = [p for p in csvs if "biosemi" in p.name.lower()]
# if not biosemi:
#     st.sidebar.error(f"❌ No BioSemi coords CSV in {configs_dir}")
#     st.stop()
# coord_csv = biosemi[0]
#
# # ─── Sidebar: select modalities & participant ─────────────────────────────────
# st.sidebar.header("Input Selection")
#
# selected_modalities = st.sidebar.multiselect(
#     "Modalities",
#     modalities,
#     default=["ECG","EEG","Pupil"]
# )
# if not selected_modalities:
#     st.sidebar.error("❌ Pick at least one modality")
#     st.stop()
#
# # auto‑discover participant IDs
# all_files = []
# for mod in selected_modalities:
#     all_files += [f.name for f in (processed_dir / mod).glob("*.csv")]
# pids = sorted({
#     re.match(r"\d+\.(P\d+)_seg", fn).group(1)
#     for fn in all_files
#     if re.match(r"\d+\.(P\d+)_seg", fn)
# })
# selected_pid = st.sidebar.selectbox("Participant ID", pids)
#
# # ─── Run Continuous Detection ─────────────────────────────────────────────────
# if st.sidebar.button("▶️ Run Continuous Detection"):
#     try:
#         # 1) Load model
#         combo_key = "_".join(selected_modalities)
#         clf       = "lgbm"
#         model_fp  = model_dir / f"{combo_key}_{clf}.pkl"
#         if not model_fp.exists():
#             raise FileNotFoundError(f"Model not found: {model_fp}")
#         model = joblib.load(model_fp)
#
#         # 2) Load top‑20 feature list
#         feat_txt = sel_txt_dir / f"{combo_key}_top20_features.txt"
#         if not feat_txt.exists():
#             raise FileNotFoundError(f"Feature list not found: {feat_txt}")
#         with open(feat_txt, "r", encoding="utf-8") as f:
#             feature_list = [ln.strip() for ln in f if ln.strip()]
#
#         # 3) Gather all segment file‐paths for this PID & each modality
#         seg_lists = []
#         for mod in selected_modalities:
#             folder = processed_dir / mod
#             files  = list(folder.glob(f"*{selected_pid}_seg*.csv"))
#             files  = sorted(
#                 files,
#                 key=lambda p: int(re.search(r"_seg(\d+)\.csv", p.name).group(1))
#             )
#             seg_lists.append(files)
#
#         # ensure same number of windows
#         lengths = {len(lst) for lst in seg_lists}
#         if len(lengths) != 1:
#             raise RuntimeError("Mismatch in segment counts across modalities")
#         n_windows = lengths.pop()
#
#         # 4) Loop: extract, filter, predict
#         fs          = config["preprocessing"]["resample_rate"]
#         window_dur  = config.get("pipeline", {}).get("window_duration_s", 5)
#         times, labels = [], []
#         X_rows = []
#
#         for idx in range(n_windows):
#             # extract full feature dict
#             feat_dict = {}
#             for mod, seg_list in zip(selected_modalities, seg_lists):
#                 df_raw = pd.read_csv(seg_list[idx])
#                 data_df = df_raw.drop(columns=["Time","label"], errors="ignore")
#                 if mod == "EEG":
#                     feats = extract_eeg_features(data_df, fs)
#                 elif mod == "ECG":
#                     feats = extract_ecg_features(data_df, fs)
#                 else:  # Pupil
#                     feats = extract_pupil_features(data_df, fs)
#                 feat_dict.update(feats)
#
#             # filter to top‑20 features only
#             filtered = {f: feat_dict[f] for f in feature_list}
#             X_rows.append(filtered)
#
#             # predict
#             X_df  = pd.DataFrame([filtered])
#             pred  = model.predict(X_df)[0]
#             labels.append(pred)
#             times.append(idx * window_dur)
#
#         # assemble results
#         df_res      = pd.DataFrame({"time_s": times, "label_idx": labels})
#         df_res["label"] = df_res["label_idx"].map({0:"Low",1:"Mid",2:"High"})
#
#         # ── Plot 1: Continuous attention ───────────────────────
#         st.subheader("Continuous Attention Over Time")
#         fig_line = go.Figure()
#         fig_line.add_trace(go.Scatter(
#             x=df_res["time_s"], y=df_res["label_idx"],
#             mode="lines+markers"
#         ))
#         fig_line.update_layout(
#             xaxis_title="Time (s)",
#             yaxis=dict(tickmode="array", tickvals=[0,1,2], ticktext=["Low","Mid","High"])
#         )
#         st.plotly_chart(fig_line, use_container_width=True)
#
#         # ── Plot 2: Gauge (last window) ───────────────────────
#         current_label = df_res["label"].iloc[-1]
#         st.subheader("Current Attention Level")
#         gauge = go.Figure(go.Indicator(
#             mode="gauge+number",
#             value=["Low","Mid","High"].index(current_label),
#             number={"suffix":f"  ({current_label})"},
#             gauge={"axis":{"range":[0,2],"tickvals":[0,1,2],"ticktext":["Low","Mid","High"]}}
#         ))
#         gauge.update_layout(height=250)
#         st.plotly_chart(gauge, use_container_width=True)
#
#         # ── Plot 3: SHAP summary ───────────────────────────────
#         st.subheader("Overall SHAP Feature Importance")
#         X_shap = pd.DataFrame(X_rows)  # shape (n_windows × 20)
#         explainer = shap.TreeExplainer(model)
#         shap_vals = explainer.shap_values(X_shap)
#         plt.figure(figsize=(6,4))
#         shap.summary_plot(shap_vals, X_shap, plot_type="dot", max_display=5, show=False)
#         st.pyplot(plt.gcf())
#
#         # ── Plot 4: EEG topomap (last window) ─────────────────
#         if "EEG" in selected_modalities:
#             st.subheader("EEG Scalp Topography (Last Window)")
#             df_last = pd.read_csv(seg_lists[selected_modalities.index("EEG")][-1])
#             ch_vals = df_last.abs().mean(axis=0)
#
#             coord_df = pd.read_csv(coord_csv, header=33, encoding="latin-1")
#             cols     = coord_df.columns
#             elec_col = [c for c in cols if "Electrode" in c][0]
#             x_col    = [c for c in cols if c.strip().startswith("x")][0]
#             y_col    = [c for c in cols if c.strip().startswith("y")][0]
#             coord_df = coord_df[[elec_col, x_col, y_col]].dropna()
#             coords   = {
#                 row[elec_col].strip(): (float(row[x_col]), float(row[y_col]))
#                 for _, row in coord_df.iterrows()
#             }
#
#             ch_present = [c for c in ch_vals.index if c in coords]
#             data  = np.array([ch_vals[c] for c in ch_present])
#             pos   = np.array([coords[c] for c in ch_present])
#
#             fig, ax = plt.subplots()
#             plot_topomap(data, pos, axes=ax, show=False)
#             ax.set_title("Mean |Voltage| per Channel")
#             st.pyplot(fig)
#         else:
#             st.info("EEG not selected → no topography")
#
#     except Exception as e:
#         st.error(f"❌ Error:\n{e}")




#============================================================================================== Code 3 ========

import os
import re
import time
from pathlib import Path

import streamlit as st
import yaml
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import mne
from mne.viz import plot_topomap

from src.features.extract_features import (
    extract_ecg_features,
    extract_eeg_features,
    extract_pupil_features,
)

# ─── Initialization ───────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🔴 Live Multimodal Attention Monitor")

# Ensure session state
if "playing" not in st.session_state:
    st.session_state.playing = False
if "cursor" not in st.session_state:
    st.session_state.cursor = 0

# ─── Sidebar: Configuration ───────────────────────────────────────────────────
st.sidebar.header("Configuration")
config_path = st.sidebar.text_input(
    "Path to config.yaml",
    value=str(Path(__file__).resolve().parent / "configs" / "config.legacy.yaml")
)
if not os.path.isfile(config_path):
    st.sidebar.error("❌ Invalid config.yaml path")
    st.stop()

# Unicode‑safe YAML load
try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
except UnicodeDecodeError:
    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)

proj_root     = Path(config["project"]["root"])
processed_dir = Path(config["data"]["processed"])
model_dir     = Path(config["models"]["output_dir"])
sel_txt_dir   = processed_dir / "selected_features"
modalities    = config["modalities"]

# BioSemi coords
coords_list = list((proj_root/"configs").glob("*biosemi*.csv"))
if not coords_list:
    st.sidebar.error("❌ Cannot find BioSemi coords CSV"); st.stop()
coords_csv = coords_list[0]

# ─── Sidebar: Inputs ──────────────────────────────────────────────────────────
sel_mods = st.sidebar.multiselect("Modalities", modalities, default=["ECG","EEG","Pupil"])
if not sel_mods:
    st.sidebar.error("❌ Pick at least one modality"); st.stop()

# flatten fused names
atomic_mods = sorted({sub for m in sel_mods for sub in m.split("_")})

# participant IDs
all_files = sum([[f.name for f in (processed_dir/m).glob("*.csv")] for m in atomic_mods], [])
pids = sorted({
    re.match(r"\d+\.(P\d+)_", fn).group(1)
    for fn in all_files
    if re.match(r"\d+\.(P\d+)_", fn)
})
if not pids:
    st.sidebar.error("❌ No participant files found"); st.stop()

pid = st.sidebar.selectbox("Participant", pids)

# Play/Pause button
if st.sidebar.button("▶️ Play" if not st.session_state.playing else "⏸️ Pause"):
    st.session_state.playing = not st.session_state.playing

# ─── Load & Concatenate Segments ──────────────────────────────────────────────
@st.cache_data
def load_continuous(mod, pid):
    folder = processed_dir / mod
    files  = sorted(folder.glob(f"*.{pid}_*.csv"), key=lambda p: int(p.name.split(".")[0]))
    if not files:
        raise FileNotFoundError(f"No files for {mod} / {pid}")
    dfs = [pd.read_csv(fp) for fp in files]
    return pd.concat(dfs, ignore_index=True)

try:
    data_stream = {m: load_continuous(m, pid) for m in atomic_mods}
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    st.stop()

# sampling rates & timeline
sr    = {m: round(1/df["Time"].diff().dropna().mean()) for m,df in data_stream.items() if "Time" in df.columns}
n_pts = max(len(df) for df in data_stream.values())
dt    = 1.0 / max(sr.values())
T     = n_pts * dt
seg_dur = config["preprocessing"]["segment_length_sec"]
seg_pts = int(sr[atomic_mods[0]] * seg_dur)

# ─── Model & Features ─────────────────────────────────────────────────────────
combo    = "_".join(sel_mods)
model_fp = model_dir / f"{combo}_lgbm.pkl"
if not model_fp.exists():
    st.error(f"❌ Model not found: {model_fp}"); st.stop()
model = joblib.load(model_fp)

feat_txt = sel_txt_dir / f"{combo}_top20_features.txt"
if not feat_txt.exists():
    st.error(f"❌ Feature list not found: {feat_txt}"); st.stop()
with open(feat_txt, "r", encoding="utf-8") as f:
    top_feats = [ln.strip() for ln in f if ln.strip()]

explainer = shap.TreeExplainer(model)

# ─── Determine current window ─────────────────────────────────────────────────
cursor = st.session_state.cursor
start  = cursor
end    = min(cursor + seg_pts, n_pts)
slice_data = {m: df.iloc[start:end] for m, df in data_stream.items()}

# ─── 1) ECG waveform ─────────────────────────────────────────────────────────
if "ECG" in slice_data:
    df_ecg = slice_data["ECG"]
    fig_ecg = go.Figure(go.Scatter(x=df_ecg["Time"], y=df_ecg.iloc[:,1]))
    fig_ecg.update_layout(title="ECG", height=200, margin=dict(t=30))
    st.plotly_chart(fig_ecg, use_container_width=True)

# ─── 2) EEG waveforms ────────────────────────────────────────────────────────
if "EEG" in slice_data:
    df_eeg = slice_data["EEG"]
    fig_eeg = go.Figure()
    for ch in df_eeg.columns.drop("Time"):
        fig_eeg.add_trace(go.Scatter(x=df_eeg["Time"], y=df_eeg[ch], name=ch))
    fig_eeg.update_layout(title="EEG Channels", height=300, margin=dict(t=30), showlegend=False)
    st.plotly_chart(fig_eeg, use_container_width=True)

# ─── 3) Pupil size & gaze ────────────────────────────────────────────────────
if "Pupil" in slice_data:
    df_pup = slice_data["Pupil"]
    fig_pup = go.Figure()
    if "r" in df_pup.columns:
        fig_pup.add_trace(go.Scatter(x=df_pup["Time"], y=df_pup["r"], name="Size"))
    if {"x","y"}.issubset(df_pup.columns):
        fig_pup.add_trace(go.Scatter(x=df_pup["x"], y=df_pup["y"], mode="markers", name="Gaze"))
    fig_pup.update_layout(title="Pupil", height=250, margin=dict(t=30))
    st.plotly_chart(fig_pup, use_container_width=True)

# ─── 4) Feature extraction & prediction ──────────────────────────────────────
feat_dict = {}
for m in atomic_mods:
    df = slice_data[m].drop(columns=["Time","label"], errors="ignore")
    if m=="ECG":    feat_dict.update(extract_ecg_features(df, sr[m]))
    elif m=="EEG":  feat_dict.update(extract_eeg_features(df, sr[m]))
    else:           feat_dict.update(extract_pupil_features(df, sr[m]))

X_row = {f: feat_dict[f] for f in top_feats}
X_df  = pd.DataFrame([X_row])
pred  = model.predict(X_df)[0]

# ─── 5) Attention gauge (colored) ─────────────────────────────────────────────
color_map = {0:"red", 1:"yellow", 2:"green"}
g = go.Figure(go.Indicator(
    mode="gauge+number",
    value=pred,
    number={"suffix": f"  ({['Low','Mid','High'][pred]})"},
    gauge={"axis":{"range":[0,2],"tickvals":[0,1,2],"ticktext":["Low","Mid","High"]},
           "bar": {"color": color_map[pred]}}
))
g.update_layout(height=250)
st.plotly_chart(g, use_container_width=True)

# ─── 6) SHAP bar chart ────────────────────────────────────────────────────────
shap_vals = explainer.shap_values(X_df)
if isinstance(shap_vals, list):
    vals = shap_vals[1] if len(shap_vals)>1 else shap_vals[0]
else:
    vals = shap_vals

st.subheader("SHAP Feature Impact (Current Window)")
plt.figure(figsize=(6,3))
shap.summary_plot(vals, X_df, plot_type="bar", max_display=5, show=False)
st.pyplot(plt.gcf())

# ─── 7) Dynamic EEG topomap ───────────────────────────────────────────────────
if "EEG" in slice_data:
    st.subheader("EEG Topography (Red–Cyan Heatmap)")
    mean_vals = slice_data["EEG"].abs().mean(axis=0).drop("Time")
    coord_df  = pd.read_csv(coords_csv, header=33, encoding="latin-1", engine="python")
    elec_col  = [c for c in coord_df.columns if "Electrode" in c][0]
    x_col     = [c for c in coord_df.columns if c.strip().startswith("x")][0]
    y_col     = [c for c in coord_df.columns if c.strip().startswith("y")][0]
    coord_df  = coord_df[[elec_col, x_col, y_col]].dropna()
    coords    = {r[elec_col]: (r[x_col], r[y_col]) for _, r in coord_df.iterrows()}

    pts  = [ch for ch in mean_vals.index if ch in coords]
    data = np.array([mean_vals[ch] for ch in pts])
    pos  = np.array([coords[ch] for ch in pts])

    fig_topo, ax = plt.subplots()
    plot_topomap(
        data, pos, axes=ax, show=False,
        cmap="RdBu_r"
    )
    ax.set_title("Mean |Voltage| per Channel")
    st.pyplot(fig_topo)

# ─── Auto‑advance if playing ──────────────────────────────────────────────────
if st.session_state.playing:
    time.sleep(seg_dur)
    new_cursor = st.session_state.cursor + seg_pts
    if new_cursor >= n_pts:
        new_cursor = 0  # loop back
    st.session_state.cursor = new_cursor
    st.rerun()

