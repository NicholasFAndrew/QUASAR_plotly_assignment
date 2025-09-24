#!/usr/bin/env python3
"""
plot_eeg_ecg.py

Load an EEG/ECG CSV (ignore lines starting with '#'), extract channels, and create
an interactive Plotly time-series plot with scrolling/zooming and simple UI toggles.

Usage:
    From Repo:
    python main.py

    Direct:
    python plot_eeg_ecg.py --input "EEG and ECG data_02_raw.csv" --output output.html --open

Dependencies:
    pandas, plotly, numpy
"""

import argparse
import io
import os
import sys
import webbrowser
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def load_eeg_ecg_csv(path: str) -> pd.DataFrame:
    """
    Load CSV while ignoring metadata lines beginning with '#'.
    The first non-comment line is used as the header row.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = []
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            # keep the first non-comment header and subsequent lines
            lines.append(line)
    if not lines:
        raise ValueError("No data found in the file after removing comment lines.")
    joined = "".join(lines)
    df = pd.read_csv(io.StringIO(joined))
    # Normalize column names by striping whitespace
    df.columns = [c.strip() for c in df.columns]
    return df


def identify_channels(df: pd.DataFrame) -> Tuple[str, List[str], List[str], Optional[str]]:
    """
    Identify Time column, EEG channels, ECG channels (X1/X2), and CM (reference).
    Construction Schema:
     - Time (seconds) column named 'Time'
     - EEG channels include Fz, Cz, P3, C3, F3, F4, C4, P4, Fp1, Fp2, T3, T4, T5, T6, O1, O2, F7, F8, A1, A2, Pz
     - ECG channels: X1:LEOG (Left ECG), X2:REOG (Right ECG)
     - CM column -> 'CM'
     - Ignore X3* and other trigger/ADC/metadata columns
    """
    colnames = df.columns.tolist()

    # Find time column
    time_candidates = ["Time", "time", "TIME", "Seconds", "seconds"]
    time_col = None
    for c in time_candidates:
        if c in colnames:
            time_col = c
            break
    if time_col is None:
        # fallback if failed to find standard time column
        for c in colnames:
            if pd.api.types.is_numeric_dtype(df[c]):
                s = df[c].dropna().values
                if s.size >= 2 and np.all(np.diff(s) >= 0):
                    time_col = c
                    break
    if time_col is None:
        raise ValueError("Could not find a suitable Time column. Please ensure the CSV has a Time column.")

    # Known EEG channel names (case-insensitive check)
    known_eeg = {
        "Fz", "Cz", "P3", "C3", "F3", "F4", "C4", "P4", "Fp1", "Fp2",
        "T3", "T4", "T5", "T6", "O1", "O2", "F7", "F8", "A1", "A2", "Pz"
    }
    eeg_cols = []
    ecg_cols = []
    cm_col = None

    for c in colnames:
        if c == time_col:
            continue
        cl = c.strip()
        # metadata to ignore
        ignore_tokens = ['Trigger', 'Time_Offset', 'ADC_Status', 'ADC_Sequence', 'Event', 'Comments']
        if any(tok == cl for tok in ignore_tokens):
            continue
        if cl.upper().startswith("X3"):  # ignore X3
            continue

        if cl == "CM":
            cm_col = cl
            continue

        # ECG detection: columns containing X1 or X2 or LEOG/REOG or containing 'ECG'
        cl_up = cl.upper()
        if "X1" in cl_up or "LEOG" in cl_up or "ECG" in cl_up and ("LE" in cl_up or "LEFT" in cl_up):
            ecg_cols.append(c)
            continue
        if "X2" in cl_up or "REOG" in cl_up or "ECG" in cl_up and ("RE" in cl_up or "RIGHT" in cl_up):
            ecg_cols.append(c)
            continue

        # known EEG names check
        if cl in known_eeg or cl.upper() in {k.upper() for k in known_eeg}:
            eeg_cols.append(c)
            continue

        # fallback heuristics:
        # if column values are small magnitude (tens-hundreds), consider EEG
        if pd.api.types.is_numeric_dtype(df[c]):
            vals = df[c].dropna().abs()
            if not vals.empty:
                median = float(vals.median())
                # EEG often tens-hundreds of uV; ECG/CM larger (thousands)
                if median < 500:
                    eeg_cols.append(c)
                else:
                    # treat large amplitude columns as possible ECG/CM if not identified
                    ecg_cols.append(c)
            else:
                # unknown empty numeric ignored
                pass
        else:
            # non-numeric columns ignored
            pass

    # Deduplicate and preserve input order
    def dedup_keep_order(lst):
        seen = set()
        out = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    eeg_cols = dedup_keep_order(eeg_cols)
    ecg_cols = dedup_keep_order(ecg_cols)

    return time_col, eeg_cols, ecg_cols, cm_col


def build_figure(df: pd.DataFrame, time_col: str, eeg_cols: List[str], ecg_cols: List[str], cm_col: Optional[str]):
    """
    Build an interactive Plotly figure with:
      - EEG traces on primary y-axis (μV)
      - ECG traces on secondary y-axis (mV) — convert from μV to mV
      - CM on secondary y-axis (mV) and dashed line
      - A range slider for the x-axis
      - Toggle buttons to show/hide EEG/ECG/CM groups
    """

    time = df[time_col]

    # Create the subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    trace_order = []
    traces = []

    # Add EEG traces (μV) on primary y-axis
    for c in eeg_cols:
        y = df[c].astype(float).values
        t = go.Scatter(
            x=time,
            y=y,
            name=f"{c} (μV)",
            mode="lines",
            line=dict(width=1),
            hovertemplate="Time: %{x}<br>%{y:.2f} μV<br><extra>%{fullData.name}</extra>",
            visible=True
        )
        fig.add_trace(t, secondary_y=False)
        traces.append(("EEG", t))
        trace_order.append(("EEG", c))

    # Add ECG traces (converted to mV) on secondary y-axis
    for c in ecg_cols:
        # convert from μV to mV (divide by 1000). If data already in mV this will just scale
        y_mV = df[c].astype(float).values / 1000.0
        t = go.Scatter(
            x=time,
            y=y_mV,
            name=f"{c} (mV)",
            mode="lines",
            line=dict(width=1),
            hovertemplate="Time: %{x}<br>%{y:.3f} mV<br><extra>%{fullData.name}</extra>",
            visible=True
        )
        fig.add_trace(t, secondary_y=True)
        traces.append(("ECG", t))
        trace_order.append(("ECG", c))

    # Add CM trace (large amplitude), plot on secondary axis (mV) with dashed line
    if cm_col:
        y_cm_mV = df[cm_col].astype(float).values / 1000.0
        t = go.Scatter(
            x=time,
            y=y_cm_mV,
            name=f"{cm_col} (mV) [CM]",
            mode="lines",
            line=dict(width=1.5, dash="dot"),
            hovertemplate="Time: %{x}<br>%{y:.2f} mV<br><extra>%{fullData.name}</extra>",
            visible=True
        )
        fig.add_trace(t, secondary_y=True)
        traces.append(("CM", t))
        trace_order.append(("CM", cm_col))

    # Layout settings
    fig.update_layout(
        title=dict(
            text="EEG & ECG Time-Series",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top'
        ),
        xaxis_title="Time (s)",
        yaxis_title="EEG (μV)",
        legend=dict(orientation="h", yanchor="bottom", y=0.95, xanchor="right", x=0.98),
        hovermode="x unified",
        margin=dict(l=70, r=20, t=80, b=20),
        xaxis=dict(rangeslider=dict(visible=True), type="linear"),
        template="plotly_white",
        height=900,
    )

    # secondary y-axis title
    fig.update_yaxes(title_text="ECG / CM (mV)", secondary_y=True)

    # Build group-visibility control buttons (EEG / ECG / CM / All / None)
    # Determine the visibility array for each button based on the traces added
    n_traces = len(fig.data)
    group_map = []
    for idx, (grp, col) in enumerate(trace_order):
        group_map.append(grp)

    def visibility_for(groups_shown: List[str]) -> List[bool]:
        return [grp in groups_shown for grp in group_map]

    # existing group buttons
    updatemenus = [
        dict(
            type="buttons",
            direction="up",
            active=0,
            x=-0.04,
            y=1.12,
            buttons=[
                dict(label="All",
                     method="update",
                     args=[{"visible": visibility_for(["EEG", "ECG", "CM"])}]),
                dict(label="EEG",
                     method="update",
                     args=[{"visible": visibility_for(["EEG"])}]),
                dict(label="ECG",
                     method="update",
                     args=[{"visible": visibility_for(["ECG"])}]),
                dict(label="CM",
                     method="update",
                     args=[{"visible": visibility_for(["CM"])}])
            ],
            showactive=True,
        )
    ]


    fig.update_layout(updatemenus=updatemenus)

    # Add a small annotation explaining the unit conversion
    annotation_text = "EEG: μV | ECG & CM plotted as mV (converted from μV)"
    fig.add_annotation(
        text=annotation_text,
        xref="paper",
        yref="paper",
        x=0,
        y=1.08,
        showarrow=False,
        align="left",
        font=dict(size=11)
    )

    return fig


def save_figure(fig: go.Figure, out_html: str):
    """
    Save interactive figure as an HTML file.
    """
    fig.write_html(out_html, include_plotlyjs="cdn", auto_open=False)
    abs_path = os.path.abspath(out_html)
    webbrowser.open(f"file://{abs_path}")
    print(f"Saved interactive HTML to: {out_html}")


def parse_args():
    p = argparse.ArgumentParser(description="Plot EEG and ECG time-series from CSV.")
    p.add_argument("--input", "-i", required=True, help="Path to CSV file.")
    p.add_argument("--output", "-o", default="eeg_ecg_plot.html", help="Output HTML filename.")
    p.add_argument("--open", action="store_true", help="Open the output HTML in a browser after saving.")
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = args.input
    out_html = args.output

    if not os.path.exists(csv_path):
        print(f"Input file does not exist: {csv_path}", file=sys.stderr)
        sys.exit(2)

    print(f"Loading data from: {csv_path}")
    df = load_eeg_ecg_csv(csv_path)
    print("Identifying channels...")
    time_col, eeg_cols, ecg_cols, cm_col = identify_channels(df)

    print(f"Time column: {time_col}")
    print(f"EEG channels detected ({len(eeg_cols)}): {eeg_cols}")
    print(f"ECG channels detected ({len(ecg_cols)}): {ecg_cols}")
    print(f"CM column: {cm_col}")

    if len(eeg_cols) == 0 and len(ecg_cols) == 0:
        print("Warning: no EEG or ECG channels identified. Check CSV header format.", file=sys.stderr)

    fig = build_figure(df, time_col, eeg_cols, ecg_cols, cm_col)

    # Save to HTML
    save_figure(fig, out_html)

    if args.open:
        # Open the saved file in the default web browser
        file_url = "file://" + os.path.abspath(out_html)
        print("Opening in default browser...")
        webbrowser.open(file_url)


if __name__ == "__main__":
    main()
