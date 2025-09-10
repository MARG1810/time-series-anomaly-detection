# Time-Series Anomaly Detection

## 📌 Overview

This project explores **anomaly detection in time-series data** using both simple statistical baselines and a deep-learning LSTM approach. The goal is to find unusual spending / fraud patterns in daily financial time-series data and compare which method finds which kinds of anomalies.

## 🎯 Problem Statement

* Detect anomalies (sudden spikes or drops) in daily spending time-series.
* Compare a rolling-statistics threshold baseline vs an LSTM autoencoder.

## 📊 Dataset

* Synthetic or real credit-card/daily-spend time-series (CSV of daily amounts).
* Preprocessed into train/test sequences for LSTM (sequence length, sliding windows).

## 🛠️ Methods

1. **Preprocessing**

   * Normalization using `MinMaxScaler` (important for LSTM stability).
   * Create fixed-length sequences for the LSTM input.

2. **Models**

   * Statistical baseline: rolling mean ± k \* rolling std (thresholding).
   * LSTM autoencoder: train to reconstruct "normal" sequences; flag high-reconstruction-error windows as anomalies.

3. **Evaluation**

   * Plot the time series with anomaly points highlighted.
   * Compare anomalies reported by the two methods and discuss differences.

## 🚀 Results

* Major spikes/drops are usually picked up by both methods.
* LSTM tends to flag subtle/contextual anomalies that simple thresholds can miss.
* Visual results (plots) are saved under `reports/figures/`.

## 📂 Project structure

```
time-series-anomaly-detection/
├── notebooks/
│   └── time_series_anomaly.ipynb
├── data/                # input CSVs (optional / ignored if large)
├── reports/figures/     # save plots here (png)
├── README.md
└── requirements.txt
```

## 🔧 Requirements

Create `requirements.txt` from Colab (example):

```bash
# in Colab notebook cell
!pip freeze > requirements.txt
```

Main libraries (examples):

* pandas, numpy, matplotlib
* scikit-learn
* tensorflow (or keras)

## ▶️ How to run

1. Clone repo:

```bash
git clone https://github.com/YOUR-USERNAME/time-series-anomaly-detection.git
cd time-series-anomaly-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook (`notebooks/time_series_anomaly.ipynb`) in Colab or Jupyter and run cells.

## 📈 Add plots to README

Save plots (in Colab):

```python
import os
os.makedirs('/content/reports/figures', exist_ok=True)
plt.savefig('/content/reports/figures/lstm_anomalies.png')
```

Upload the file to `reports/figures/` in your repo (GitHub web UI → Add file → Upload files) and embed in README:

markdown
![LSTM anomalies](reports/figures/lstm_anomalies.png)


## 🔍 Next steps

* Try other models (Prophet, ARIMA).
* Add a small Flask/FastAPI wrapper to serve anomaly scores.
* Test on real-world financial datasets.



Created by: Margi Dave
