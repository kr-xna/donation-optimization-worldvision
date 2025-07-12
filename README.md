# Predictive Modeling for Donation Optimization

This project was developed as part of my Master's in Business Analytics at Deakin University in collaboration with World Vision Australia. The objective was to predict donor contributions and optimize outreach efforts to improve donor engagement.

## Problem Statement

World Vision sought to improve the effectiveness of their donor campaigns by understanding key factors influencing donation amounts. The goal was to build predictive models that estimate `TotalPaid` (donation amount) based on supporter demographics, interactions, and engagement data.

## Tools & Technologies

- **Languages & Libraries:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Techniques:**
  - Data cleaning, sampling, and merging multi-source datasets
  - Feature engineering (e.g., age, campaign length, engagement duration)
  - One-hot encoding, missing value imputation
  - Model building: Linear Regression, Decision Tree, Random Forest, Gradient Boosting
  - Feature importance & selection
  - Visualization of model performance and correlations

## Approach

- Integrated multiple datasets (`Responses`, `Demographics`, `Audiences`, `Contacted`, etc.) using `Deakin_SupporterID`
- Performed feature engineering to derive variables such as age, campaign length, engagement duration, and interaction frequency
- Handled missing data and applied sampling for computational efficiency
- Built and evaluated several regression models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Visualized predictions against actual values and analyzed feature importance

## Results

- The best-performing model (e.g., Random Forest) demonstrated strong predictive capability on the test data
- Identified key predictors of donation amount, providing actionable insights for targeting high-value donors
- Delivered recommendations to the World Vision team for campaign optimization

## Visualizations

- Donation trends over time
- Product type and demographic distributions
- Correlation heatmaps
- Predicted vs. actual plots for each model

## Key Learnings

- The importance of feature engineering in improving predictive accuracy
- Comparison of ensemble methods with simpler models
- Handling real-world, messy data with missing values and inconsistent formats

## Contact

- Email: [rkrishnamanirao@gmail.com](mailto:rkrishnamanirao@gmail.com)
- LinkedIn: [linkedin.com/in/krishnamanirao](https://linkedin.com/in/krishnamanirao)

*"Data tells stories — I help narrate them."*
