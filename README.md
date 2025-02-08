# Oil Price Prediction Using Transformers & Sentiment Analysis

## Overview
This project is a **full-stack data science pipeline** for forecasting oil prices using **state-of-the-art (SOTA) transformer models**. The approach integrates:  
- **Economic and geopolitical sentiment analysis**, extracted from global news sources using LLMs.  
- **Deep learning and statistical forecasting models**, benchmarked for performance.  
- **HPC and GPU-powered training**, optimizing hyperparameters for maximum accuracy.  

Our goal was to improve oil price prediction by incorporating sentiment-driven signals alongside traditional economic indicators.  

---

## Key Features
- **End-to-End Development**: From data collection to hyperparameter tuning, all on **HPC clusters with GPUs**.  
- **SOTA Transformer Models**: Used **PatchTST** and **Temporal Fusion Transformer (TFT)** for time-series forecasting.  
- **Sentiment Analysis from News Sources**:  
  - Scraped **9 financial and geopolitical news sources** (including Middle Eastern outlets) using **Selenium**.  
  - Extracted sentiment using **Mistral, LLAMA, and HuggingFace Twitter BERT**.  
- **20 Features Engineered for Forecasting**, including economic indicators, sentiment scores, and time-based variables.  
- **Benchmarking Against Traditional Models**: Compared results with **ARIMAX and SARIMA**.  
- **Hyperparameter Optimization**: Used **Optuna** to fine-tune deep learning models.  
- **Performance Tracking**: Logged results using **TensorBoard**.  

---

## Data & Feature Engineering
The dataset comprises **economic indicators, sentiment signals, and temporal features**, ensuring a robust feature set for forecasting.  

### 1. Economic Variables (8 Features)
These indicators capture **macroeconomic** and **market-driven** effects on oil prices:  
- **S&P 500** – Tracks overall market sentiment and economic health.  
- **Interest Rates** – Affects investment, consumption, and energy demand.  
- **Oil Futures** – Predicts future oil prices based on contract pricing.  
- **VIX (Volatility Index)** – Measures broader market uncertainty.  
- **OVX (Oil Volatility Index)** – Indicates crude oil market volatility.  
- **USO (United States Oil Fund ETF)** – Tracks oil futures, acting as a crude price proxy.  
- **DXY (U.S. Dollar Index)** – Impacts oil prices as oil is traded in USD.  
- **Crude Oil Spot Prices** – Real-time market price, serving as a key predictor.  

### 2. Sentiment Features (4 Features)
Extracted using **Mistral, LLAMA, and BERT** from **global financial news sources**:  
- **Global Market Sentiment** – Overall sentiment from financial and economic news.  
- **Middle East Geopolitical Sentiment** – Captures regional tensions affecting oil supply.  
- **U.S. Economic Sentiment** – Extracted from U.S. news on policies, inflation, and economic outlook.  
- **Oil-Specific Sentiment** – Direct mentions of oil prices and supply-demand dynamics.  

### 3. Engineered Temporal Features (8 Features)
Designed for **seasonality modeling** and capturing long-term dependencies:  
- **Sin(Date) & Cos(Date)** – Encodes cyclic seasonal patterns.  
- **Target Lag (7 Days, 30 Days, 365 Days)** – Captures weekly, monthly, and annual trends.  
- **Day of the Week** – Accounts for weekday effects on oil trading.  
- **Month of the Year** – Helps capture seasonality in oil prices.  
- **Holiday Indicator** – Flags major holidays affecting market activity.  

---

## Modeling Approach
### 1. Deep Learning Models
We trained two transformer-based models:  
- **PatchTST** – A SOTA transformer architecture optimized for time-series forecasting.  
- **Temporal Fusion Transformer (TFT)** – Captures complex dependencies in multivariate time series.  

### 2. Traditional Statistical Models (Benchmarks)
For comparison, we evaluated:  
- **ARIMAX** – Time-series regression with external variables.  
- **SARIMA** – Seasonal Auto-Regressive Integrated Moving Average.  

### 3. Hyperparameter Optimization
- **Optuna** was used for automated hyperparameter tuning.  
- Models were trained on **HPC clusters with GPUs** for faster processing.  

---

## Tech Stack & Libraries
We leveraged a combination of deep learning, forecasting, and scraping tools:  

### Deep Learning & Forecasting
- `nixtla` – Advanced time-series forecasting library.  
- `pytorch-forecasting` – Deep learning models for time-series data.  
- `fastai`, `tai` – Accelerated deep learning frameworks.  

### Traditional Forecasting Models
- `statsforecast` – Efficient implementation of ARIMA/SARIMA models.  

### Sentiment Analysis & Web Scraping
- `selenium` – Automated news scraping from global sources.  
- `Hugging Face Transformers` – Used for sentiment extraction with Mistral, LLAMA, and BERT.  

### Optimization & Experiment Tracking
- `optuna` – Hyperparameter tuning.  
- `tensorboard` – Performance logging and visualization.  

---

## Results & Insights
- **Transformers Outperformed Traditional Models**: PatchTST and TFT captured **complex dependencies** better than ARIMAX/SARIMA.  
- **Sentiment-Driven Signals Improved Accuracy**: Incorporating geopolitical and economic sentiment **enhanced model predictive power**.  
- **Hyperparameter Tuning Boosted Model Performance**: Optuna-led optimizations significantly **reduced forecasting errors**.  

---

## How to Run the Project
### 1. Set Up Environment
Install dependencies:  
```bash
pip install -r requirements.txt
