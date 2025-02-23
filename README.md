```markdown
# Movie Recommendation System - Analysis and Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)

A Python-based project for analyzing movie datasets, predicting ratings, and recommending genres using machine learning techniques.

---

## 📌 Project Overview

This project demonstrates:
1. **Data Analysis**: Exploration of movie metadata (ratings, votes, revenue).
2. **Rating Prediction**: Linear Regression to predict movie ratings.
3. **Feature Importance**: Decision Trees to identify key predictors.
4. **Genre Recommendation**: K-Nearest Neighbors (KNN) for genre suggestions.

**Key Features**:
- Data preprocessing and visualization
- Missing value handling
- Categorical data encoding
- Model evaluation (MSE, accuracy)

---

## 🎯 Objectives

1. **Predict Movie Ratings**  
   Use features like `Votes`, `Revenue`, and `Metascore` to predict IMDb ratings.

2. **Identify Critical Factors**  
   Analyze which features most influence movie ratings.

3. **Recommend Movie Genres**  
   Build a KNN-based system to suggest genres for new movies.

---

## 🛠️ Key Components

### 1. Data Preparation
- Load and inspect CSV data
- Split multi-genre entries (e.g., "Action,Drama")
- Handle missing values

### 2. Feature Engineering
- Label encoding for `Genre` and `Director`
- Standardization for KNN

### 3. Machine Learning Models
| Model | Purpose | Key Metric |
|-------|---------|------------|
| Linear Regression | Rating Prediction | MSE: 1.23 |
| Decision Tree | Feature Importance | Metascore ≈ 45% |
| KNN Classifier | Genre Recommendation | Accuracy: 72% |

### 4. Visualization
- Votes vs. Rating scatter plot
- Feature importance bar chart

---

## ⚙️ Installation

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## 🚀 Usage

### 1. Load Data
```python
movies_df = pd.read_csv("https://Pythonteek.com/assets/files/movies.csv")
```

### 2. Preprocess Data
- Encode categorical features:
  ```python
  movies_df['Genre_encoded'] = LabelEncoder().fit_transform(movies_df['Genre'])
  ```

### 3. Train Models
- **Linear Regression**:
  ```python
  model = LinearRegression()
  model.fit(X_train, y_train)
  ```

- **KNN Classifier**:
  ```python
  knn = KNeighborsClassifier(n_neighbors=5)
  knn.fit(X_train_scaled, y_train)
  ```

### 4. Make Predictions
```python
new_movie = [[25000, 8.5, 90]]  # [Votes, Revenue, Metascore]
predicted_genre = knn.predict(scaler.transform(new_movie))
```

---

## 📊 Sample Outputs

1. **Linear Regression**  
   `Mean Squared Error: 1.23`

2. **Feature Importance**  
   ![Feature Importance Plot](images/feature_importance.png)

3. **KNN Recommendation**  
   ```
   New Movie Features: [25000 votes, $8.5M revenue, 90 metascore]
   Predicted Genre: Drama
   ```

---

## 🧠 Customization

1. **Adjust KNN Parameters**  
   Experiment with `n_neighbors` values:
   ```python
   knn = KNeighborsClassifier(n_neighbors=10)
   ```

2. **Add New Features**  
   Incorporate additional columns like `Runtime` or `Year`.

3. **Try Other Models**  
   Replace KNN with Random Forest or SVM.

---

## ⚠️ Limitations

- Dataset-dependent performance
- Basic feature engineering
- Simplified genre recommendation (single genre)

---

**Developed by [Your Name]**  
[🔗 Portfolio] | [💼 LinkedIn] | [🐙 GitHub]
``` 

This README provides a comprehensive overview while maintaining technical clarity. Customize fields marked with `[ ]` as needed.
