# ml-gold-silver-price-direction-EDA
An EDA and Baseline Modeling project using CRISP-DM to predict the daily directional movement of Gold and Silver prices using macro-economic indicators.

## Business Understanding
### Objective
The primary goal of this task is to determine if macroeconomic indicators (such as USD Index, S&P 500, Interest Rates, Inflation, Oil & Natural Gas Price, US Equity Market Volatility) and historical price movements of precious metals can be leveraged to predict the daily price direction of Gold and Silver.

### Problem
Gold and silver play important role mainly as hedging instruments during periods of inflation, economic uncertainty. Without clear understanding of how macroeconomic factors influence these metals, investors will have incomplete information leading to ineffective decisions. This project translates complex economic data into actionable business intelligence that supports better investment strategies and portfolio diversification.

### Research Question
How do key macroeconomic indicators such as USD Index, S&P 500, Interest Rates, Inflation, Oil & Natural Gas Price, US Equity Market Volatility influence the price movements of precious metals gold and silver ?
Which macroecononic indicators have highest influence on gold and silver price movements.
Is silver more accurately predictable than gold ?

## Data Understanding

### Data Description

Following data will be used for gold/silver price direction prediction which is comprised of historical daily observations spanning from 2000 to 2026, integrated from multiple sources:

**Target Variables (Gold and Silver)**:

* **Gold (GC=F)**: Daily closing price in USD per troy ounce.
* **Silver (SI=F)**: Daily closing price in USD per troy ounce.
  
**Energy Commodities**:

* **WTI Crude Oil (CL=F)**: The benchmark for U.S. oil prices, it serves as a leading indicator for global industrial demand and energy-driven inflation.
* **Natural Gas (NG=F)**: A key energy input and highly volatile commodity that influences broader commodity market sentiment.

**Currency & Equity Benchmarks**:

* **U.S. Dollar Index (DTWEXBGS)**: It is a measure of the dollar's value against a basket of six major foreign currencies (Euro, Japanese Yen, British Pound, Canadian Dollar, Swedish Krona, Swiss Franc) to gauge its strength in global markets. Historically, USD Index exhibits a strong inverse correlation with Gold.
* **S&P 500 Index (^GSPC)**: A benchmark for the U.S. equity market, measures the stock prices of 500 of the largest, publicly-traded companies in the US.

**Interest Rates & Volatility**:

* **10-Year Treasury Yield (^TNX)**: The annual return an investor earns on a U.S. Treasury note that matures in 10 years. The return on benchmark U.S. government debt.
* **CBOE Volatility Index (VIX)**: Often called the "Fear Index", it measures market expectations of near-term volatility.

**Inflation Indicators (Macro)**:

* **Consumer Price Index (CPI)**: Monthly index level transformed into YoY and MoM growth rates. Used to capture the inflationary regime, a fundamental driver of "Safe Haven" asset demand.

### Data Acquisition & Merging
To ensure high quality and time-series data, it has been downloaded from Yahoo Finance and FRED (Federal Reserve Economic Data) data sources based on following tickers using python modules - yfinance and pandas_datareader.

* **GC=F** : Gold Futures (Yahoo Finance)
* **SI=F** : Silver Futures (Yahoo Finance)
* **CL=F**: WTI Oil Futures (Yahoo Finance)
* **NG=F**: Natural Gas Futures (Yahoo Finance)
* **^GSPC** : S&P 500 Index (Yahoo Finance)
* **^TNX** : CBOE 10-Year Treasury Note Yield Index (Yahoo Finance)
* **^VIX** : CBOE Volatility Index (Yahoo Finance)
* **DFII10** : Real Interest Rates - Inflation-Indexed (FRED)
* **DTWEXBGS** : USD Index (FRED)
* **CPIAUCSL** : Consumer Price Index Data (FRED)

- The raw data acquired is from year 2000 for all of the metrics except USD Index which is starting from year 2006. All except, CPI, is on daily basis.
- CPI data is released on monthly basis. So it creates mismatch with other data on daily basis. So the CPI index will be expanded from monthly into daily basis by forward filling. However, calculating daily percentage changes on upsampled data would result in signal sparsity, where the feature contains zero values for approximately 20 out of 21 trading days per month. So the CPI month-on-month (MoM) and year-on-year (YoY) values are computed while data is still in original format before the CPI index will be expanded using forward filling method.
- After acquiring raw data, all the datasets have been merged to produce a single dataset indexed by date.

### Data Quality Checks

- The data contains few missing values, mainly due to merging of the datasets. The usd_index has high number of missing values because it's data is available beginning 2006 while other data is available beginning 2000. During data preparation phase, the missing values will be handled.
- There are no duplicates
- All the features are represented as floating point numbers and will be retained as the same data type to preserve the precision.
- Dataset contains 6554 rows with unique dates, with data from Jan 2000 to Jan 2026. The data is very consistent covering each weekday (except weekends).

### Exploratory Data Analysis (EDA)

Several visualization were created to understand the data, its anomolies. Below are the key insights.

- Gold prices show a long-term upward trend with notable spikes during periods of economic uncertainty such as the 2008 financial crisis and the COVID-19 pandemic.
- Silver prices exhibit higher volatility compared to gold.
- Gold-Silver ratio tends to be within certain range indicating that ratio is mostly stationary and it also shows that Gold outperformed Silver during periods of economic uncertainity.
- Gold-Silver ratio can play an important role in the price prediction, it will be included as a feature in the dataset for model training
- As Gold and silver prices exhibit non-normal distributions, we will explore tree-based algorithms during modeling phase.
- Gold price distribution is much wider compared to that of silver, reflecting substantial long-term price appreciation and sensitivity to macroeconomic events such as financial crises, inflationary periods.
- Silver price distribution is more compact and seems more volatile in short-term compared to gold prices.
- Price and Index levels: Visual analysis of price and index levels (Gold, Silver, S&P 500, USD) showed significant long-term movements and non-stationarity. These must be converted to daily percentage returns to ensure the model learns from relative movement rather than absolute price levels.
- Volatility: Volatility indicators like the VIX were observed to be "mean-reverting" and range-bound. These features provide a constant measure of market fear and can be used as-is without any further derivations.
- Interest Rates: The 10-year yield and real interest rate moves in absolute basis points, so calculating the daily absolute change captures the velocity of interest rate changes.
- Inflation: To resolve the monthly-to-daily frequency mismatch, CPI YoY and MoM growth rates are used to provide a steady "inflationary regime" signal, as daily changes in monthly data would result in no signal for most of the days.
- Correlation analysis of raw price levels of Gold and USD Index showed strong positive relationship between these two features which is not in accordance to the economic theory of inverse relationship, as both Gold and USD Index exhibits long-term drift causing spurious correlation. To resolves this, features were transformed into daily percent changes (returns) which correctly reflects the expected relationships. Same theory has been applied to other features wherever necessary.
- Exploratory analysis of WTI Crude Oil revealed extreme outliers during the April 2020 oil crash where prices dropped to -$37.63 per barrel due to storage capacity exhaustion and contract expirations.This extreme outlier may influence prediction signals incorrectly, so to prevent these rare observations to skew the model's coefficients, the price was capped at 0.01 (instead of deleting to preserve the rest of the features for that day).
- Although gold and silver prices seems to be related, they exhibit different sensitivities to macroeconomic indicators. Therefore, separate predictive models will be developed for gold and silver to capture their distinct market dynamics and to provide clearer, metal-specific insights.
- Correlation between 10y_yield_chg and real_rate_chg (0.78) is high, but it is still less than the higher bound for statistical analysis which is 0.90. For now, we will keep both the features. During feature importance evaluation, we may exclude one of these features as needed.

## Data Preparation

- Extreme Outliers: Clip the extreme outlier (negative) price for WTI oil to $0.01
- Feature Transformations: Add new features for keeping percent or absolute daily changes to prevent spurious correlation
- Feature Engineering
  - Add gold/silver ratio
  - 1-day lag is applied to all independent variables to ensure the model uses only historically available information, thereby preventing data leakage.
  - In addition to the primary feature lags, a suite of technical features such as 5-day momentum, distance from the 20-day SMA, and 10-day rolling volatility were engineered. While the baseline model will be trained with lag features, these new features are prepared for the Modeling and Evaluation phase, where they will be used to test if capturing market trend and volatility improves predictive accuracy over the baseline.
  - Add target variables for gold and silver which basically contains next day's price as the model need to predict next day's price based on the current day's values.
 
## Modeling





