# main.py
import sys
import pandas as pd
from plot_eeg_ecg import build_figure, save_figure   # functions

if __name__ == "__main__":
    # default to EEG_and_ECG_data_02_raw.csv if no argument given
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "EEG and ECG data_02_raw.csv"
    df = pd.read_csv(csv_path, comment="#")

    fig = build_figure(df,
                       time_col="Time",
                       eeg_cols=["Fz", "Cz", "P3", "C3", "F3", "F4", "C4", "P4", "Fp1", "Fp2",
        "T3", "T4", "T5", "T6", "O1", "O2", "F7", "F8", "A1", "A2", "Pz"],  # list of EEG columns
                       ecg_cols=["X1:LEOG","X2:REOG"],  # list of ECG columns
                       cm_col="CM")     # or None

    save_figure(fig, "output.html")
    print("Open output.html in your browser.")