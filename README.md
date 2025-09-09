# **Project Name**    - Yes Bank Stock Closing Price Prediction

## **1. Problem Statement**

Predict the closing price of Yes Bank stock based on historical data. This can be framed as a regression problem, where we aim to predict a continuous target variable (closing price).

## **2. Business Context**

*   **Investment Strategy:** Predicting stock prices can inform investment decisions, allowing investors to buy or sell stocks based on expected price movements.
*   **Risk Management:** Understanding stock price volatility and trends can help manage risk associated with investments.
*   **Financial Forecasting:** Accurate stock price predictions can contribute to broader financial forecasting and market analysis.

## **3. Data Understanding**

*   **Dataset:** The dataset contains historical daily stock prices for Yes Bank, including the date, opening price, high price, low price, and closing price.
*   **Features:**
    *   Date: The date of the stock price observation.
    *   Open: The opening price of the stock on that day.
    *   High: The highest price reached during the trading day.
    *   Low: The lowest price reached during the trading day.
    *   Close: The closing price of the stock on that day (target variable).
*   **Data Source:** The data source is assumed to be reliable and publicly available.

## **4. Dataset Loading and Cleanup**

The dataset was loaded using pandas and mounted from Google Drive. Missing values were checked and confirmed to be zero. The 'Date' column was converted to datetime objects.

## **5. Exploratory Data Analysis (EDA)**

*   **Dataset Shape:** The dataset contains 185 rows and 5 columns.
*   **Descriptive Statistics:** Summary statistics were generated for the numerical features (Open, High, Low, Close), showing the count, mean, min, max, and quartiles.
*   **Visualizations:**
    *   **Line Plot - Closing price over time:** A line plot of 'Close' vs 'Date' was generated to visualize the stock price trend over time.
    *   **Histograms:** Histograms were created for 'Open', 'High', 'Low', and 'Close' to visualize their distributions.
    *   **Box Plots:** Box plots were generated for 'Open', 'High', 'Low', and 'Close' to identify potential outliers and the spread of the data.
    *   **Scatter Plots:** Scatter plots were created to visualize the relationships between numerical features, including 'Open' vs. 'Close' and a scatter plot matrix of all numerical features.

**Patterns and Trends:**

*   The line plot shows the general trend of the Yes Bank stock price over time, indicating periods of fluctuation.
*   Histograms and box plots provide insights into the distribution and presence of outliers in the stock price features.
*   Scatter plots reveal strong positive correlations between 'Open', 'High', 'Low', and 'Close' prices.

**Outliers:**

*   Box plots show outliers in the 'Open', 'High', 'Low', and 'Close' features, indicating extreme values.

**Relationships between Features:**

*   Strong positive correlations were observed between 'Open', 'High', 'Low', and 'Close' prices, as confirmed by the scatter plots and correlation matrix.

## **6. Feature Engineering**

**Encoding:** No categorical features were present in the dataset.

**Feature Creation:**

New features were created to enhance the dataset for modeling:

*   **Daily Percentage Change:** Calculated as `dataset['Close'].pct_change()`.
*   **Moving Averages:** 5-day (`MA5`) and 20-day (`MA20`) moving averages of the closing price were calculated.
*   **High-Low Difference:** Calculated as the difference between 'High' and 'Low' prices.

Missing values introduced by calculating moving averages were filled with the mean of the respective columns.

**Multicollinearity Handling:**

A correlation matrix was calculated to identify highly correlated features. Strong positive correlations were observed between 'Open', 'High', 'Low', and 'Close'. While multicollinearity exists, these features are highly relevant for stock price prediction. The decision was made to keep these features for the initial modeling phase.

## **7. Target Feature Conditioning**

**Checking for Skewness:**

The distribution of the target variable, 'Close', was visualized using a histogram with a KDE curve. The skewness was calculated, resulting in a value of 1.25, indicating a moderate right-skew.

**Applying Transformations:**

A log transformation was applied to the 'Close' price, creating a new feature 'LogClose'. The distribution of 'LogClose' was then visualized, and the skewness was recalculated, resulting in a value of -0.027. This value is very close to 0, indicating that the log transformation successfully made the target variable's distribution close to normal.

## **8. Model Implementation**

**1. Train-Test Split:**

The dataset was split into training (80%) and testing (20%) sets using `train_test_split` with `random_state=42`. The features used were 'Open', 'High', 'Low', 'MA5', 'MA20', and 'High_Low_Diff', and the target variable was the transformed 'LogClose'.

**2. Model Fitting:**

Two regression models were implemented:

*   **Linear Regression:** A standard Linear Regression model was trained.
*   **Random Forest:** A Random Forest Regressor model was trained with `random_state=42`.

**3. Testing and Evaluation:**

The models were evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared.

*   **Linear Regression:** MSE: 0.1452, RMSE: 0.3811, R-squared: 0.8231
*   **Random Forest:** MSE: 0.0325, RMSE: 0.1803, R-squared: 0.9604

Interpretation: Random Forest significantly outperformed Linear Regression based on these metrics.

**4. Regularization (for Linear Regression):**

Ridge and Lasso regression models were also implemented to address potential multicollinearity and improve Linear Regression performance.

*   **Ridge Regression:** MSE: 0.1452, RMSE: 0.3811, R-squared: 0.8231
*   **Lasso Regression:** MSE: 0.1343, RMSE: 0.3665, R-squared: 0.8364

Interpretation: Lasso Regression slightly outperformed Ridge Regression.

**Hyperparameter Tuning:**

Hyperparameter tuning was performed using Grid Search for Ridge and Lasso Regression (with feature scaling) and Randomized Search for Random Forest.

*   **Ridge Regression (Tuned):** MSE: 0.1451, RMSE: 0.3809, R-squared: 0.8233
*   **Lasso Regression (Tuned):** MSE: 0.1333, RMSE: 0.3650, R-squared: 0.8377
*   **Random Forest (Tuned with Random Search):** MSE: 0.0285, RMSE: 0.1688, R-squared: 0.9653

Interpretation: Hyperparameter tuning further improved the performance of all models, with Random Forest maintaining its superior performance.

## **9. Model Explainability**

**Feature Importance:**

Feature importance was analyzed for both Linear Regression (using coefficients) and Random Forest (using `feature_importances_`).

*   **Linear Regression:** 'High', 'Low', and 'Open' were the most important features, followed by 'High_Low_Diff', 'MA20', and 'MA5'.
*   **Random Forest:** 'Low' and 'High' were the most important features, followed by 'Open', 'MA5', 'MA20', and 'High_Low_Diff'.

**Key Takeaways:**

Both models identified 'High', 'Low', and 'Open' prices as the most influential predictors of the closing price. Moving averages and the daily price range had less influence.

## **10.Conclusion**

This project successfully developed a model for predicting Yes Bank stock prices using historical data and machine learning. The Random Forest model, after hyperparameter tuning, demonstrated excellent performance with a low MSE and RMSE and a high R-squared value (0.9653), indicating its strong predictive capability. Feature importance analysis highlighted the significance of 'Low', 'High', and 'Open' prices in predicting the closing price.

**Key Findings**

*   Exploratory data analysis revealed trends, patterns, and relationships in the data.
*   Feature engineering improved the model's ability to capture relevant information.
*   Random Forest consistently outperformed Linear Regression models.
*   Hyperparameter tuning further enhanced model performance.
*   'Low' and 'High' prices were the most influential predictors.

**Overall**

The developed Random Forest model shows promising predictive capabilities for Yes Bank stock prices and could be valuable for informed decision-making.

**Future Directions**

*   Incorporate external factors (news sentiment, economic indicators).
*   Explore advanced models (deep learning, time series models).
*   Develop capabilities for real-time prediction.
*   Integrate risk management and portfolio optimization strategies.

**Disclaimer:**

Stock market prediction is complex and uncertain. This model is a research prototype and should not be the sole basis for investment decisions. Always conduct thorough due diligence and seek professional financial advice.

By continuing to refine and enhance this model, while acknowledging the inherent risks associated with financial markets, we can strive towards more robust and reliable stock price prediction tools.

## **Github Link -** https://github.com/KRamBalaji/stock_closing_price_prediction
