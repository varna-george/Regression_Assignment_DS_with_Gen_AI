Regression Assignment

Regression Assignment Objective:
The objective of this assignment is to evaluate the understanding of regression techniques in supervised learning by applying them to a real-world dataset.

Dataset:
Used California Housing dataset available in the sklearn library. This dataset contains information about various features of houses in California and their respective median prices.

Key Components fulfilled:
1. Loading and Preprocessing:
● Loaded the California Housing dataset using the fetch_california_housing function from sklearn.
● Converted the dataset into a pandas DataFrame for easier handling.
● All the features are float and there are no categorical feature.
● No missing values found. 
● Scaling is done only for Linear Regression and SVR
Linear Regression: Scaling (e.g., StandardScaler) makes sure all features are on the same scale, so the model treats them equally.
Support Vector Regressor (SVR): Scaling ensures that each feature contributes proportionally to the distance measure.
Decision Tree, Random Forest, Gradient Boosting: These models split data by feature thresholds, scaling has no effect on their performance.

2. Regression Algorithm Implementation:
Implemented the following regression algorithms:
○ Linear Regression
Linear Regression models the relationship between the target variable y and input features X as a straight line (a linear combination).
It tries to find the best-fitting line that minimizes the sum of squared errors between predicted and actual values.
California Housing dataset contains non-linear relationships (e.g., prices may rise quickly with income up to a point, then level off).
Hence, Linear Regression gives reasonable but not top performance (R² ≈ 0.58).

○ Decision Tree Regressor
Splits the data recursively into smaller regions based on feature thresholds that reduce prediction error the most.
Each terminal node (leaf) represents the predicted target value for samples in that region.
Captures non-linear relationships and feature interactions automatically.
Handles both large and small scales of data easily (no need for scaling).
In this dataset, it performs better than Linear Regression, but it can overfit if not tuned (R² ≈ 0.62).

○ Random Forest Regressor
An ensemble of many Decision Trees trained on random subsets of data and features.
The final prediction is the average of all trees’ predictions, reducing overfitting and variance.
Works exceptionally well for tabular, mixed-scale, and non-linear data like this dataset.
It captures complex relationships between features such as MedInc, AveRooms, and Population.
Achieves highest accuracy (R² ≈ 0.81) because it generalizes well while maintaining low variance.

○ Gradient Boosting Regressor
Builds an ensemble sequentially, where each new tree tries to correct the errors of the previous one.
Combines weak learners (shallow trees) into a strong predictive model through gradient-based optimization.
Excellent for non-linear and complex datasets like California Housing.
Learns patterns iteratively and fine-tunes predictions.
Performs slightly below Random Forest here (R² ≈ 0.78), but with proper tuning (learning rate, n_estimators), it can even surpass it.

○ Support Vector Regressor (SVR)
Uses the concept of maximum margin from Support Vector Machines.
Can use kernels (like RBF) to capture non-linear relationships.
Needs feature scaling and can be slow for large datasets like California Housing (~20,000 rows).
Performs decently (R² ≈ 0.73), showing it captures non-linearity but not as efficiently as ensemble trees.

3. Models Evaluation and Comparison:
Result
               Model       MAE       MSE      RMSE        R2
0      Random Forest  0.327543  0.255368  0.505340  0.805123
1  Gradient Boosting  0.371643  0.293997  0.542215  0.775645
2                SVR  0.398599  0.357004  0.597498  0.727563
3      Decision Tree  0.454679  0.495235  0.703729  0.622076
4  Linear Regression  0.533200  0.555892  0.745581  0.575788

Random Forest performs the best:
*Lowest errors (MAE, MSE, RMSE)
*Highest R² (0.805) → explains ~80% of variance
*Benefits from ensembling multiple trees → reduces overfitting compared to a single Decision Tree

Gradient Boosting is the second-best model:
*Slightly higher error than Random Forest, but still strong

SVR performs decently but worse than ensemble trees:
*Sensitive to feature scaling and kernel choice
*Slower to train on larger datasets like California Housing

Decision Tree:
*Lower performance than ensemble models
*Can overfit easily

Linear Regression:
*Lowest performance
*Cannot capture non-linear relationships in the data
*Simple, interpretable model, but less predictive power here
