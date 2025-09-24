# QUASAR Plotly Assignment

A short project for the GUI Development Intern role at QUASAR.  
Uses **Python**, **Plotly**, and **Pandas** to generate an interactive graph and embed it in an HTML file.

---

## Features

- Plots **EEG**, **ECG**, and **CM** time-series data from a CSV file.
- EEG plotted in **µV** (primary y-axis).
- ECG and CM converted to and plotted in **mV** (secondary y-axis).
- Interactive controls:
  - Toggle traces on/off directly from the legend.
  - Pan, zoom, and reset view.
  - Range slider for navigating long time-series.
- Exports to a **self-contained HTML file** that opens in any browser (no dependencies required once generated).

---

## Development

  If allowed more time, I would have liked to add a better method of toggling on/off which data streams are
  being rendered though the use of a sorted grid of checkboxes, though that would have required more extensive
  HTML configuration. 

  LLMs such as ChatGPT were used to fill in areas of knowledge due to unfamiliar medium of GUI 
  development. Conceptualization and creation of the structure and design were done manually.

---

## How to Run

-Cloning

  git clone https://github.com/NicholasFAndrew/QUASAR_plotly_assignment.git
  
  cd QUASAR_plotly_assignment

-Requirements

  pip install -r requirements.txt

  Python **3.12+**
  numpy>=1.23,<2.0
  pandas>=1.5,<3.0
  plotly>=5.15,<6.0

-Running

  python main.py

  **OR**

  python plot_eeg_ecg.py --input "EEG and ECG data_02_raw.csv" --output output.html --open
