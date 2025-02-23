import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier

# خواندن فایل CSV
file_path = "https://Pythonteek.com/assets/files/movies.csv"
movies_df = pd.read_csv(file_path)

# نمایش ۵ سطر اول
print("First 5 rows:")
print(movies_df.head())
print("\nColumns:", movies_df.columns)

# جداسازی ژانرها
genres_split = movies_df['Genre'].str.split(',', expand=True)
movies_df = pd.concat([movies_df, genres_split], axis=1)
print("\nData after splitting genres:")
print(movies_df.head(3))

# مدیریت مقادیر گمشده
movies_df.dropna(subset=['Votes', 'Revenue', 'Metascore'], inplace=True)

# تبدیل داده‌های متنی به عددی با encoderهای جداگانه
label_encoder_genre = LabelEncoder()
label_encoder_director = LabelEncoder()

movies_df['Genre_encoded'] = label_encoder_genre.fit_transform(movies_df['Genre'])
movies_df['Director_encoded'] = label_encoder_director.fit_transform(movies_df['Director'])

print("\nEncoded values:")
print(movies_df[['Genre', 'Genre_encoded', 'Director', 'Director_encoded']].sample(5))

# رسم نمودار Votes vs Rating
plt.figure(figsize=(10, 6))
plt.scatter(movies_df['Votes'], movies_df['Rating'], alpha=0.5)
plt.xlabel('Votes')
plt.ylabel('Rating')
plt.title('Votes vs. Rating')
plt.show()

# بخش رگرسیون خطی
X = movies_df[['Votes', 'Revenue', 'Metascore']]
y = movies_df['Rating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# آموزش مدل رگرسیون
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# ارزیابی مدل
y_pred = lin_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nLinear Regression MSE: {mse:.2f}")

# تحلیل اهمیت ویژگی‌ها با درخت تصمیم
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

plt.figure(figsize=(10, 6))
plt.bar(X.columns, tree_model.feature_importances_)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances (Decision Tree)')
plt.show()

# بخش سیستم پیشنهاددهنده با KNN
# آماده‌سازی داده‌ها
X = movies_df[['Votes', 'Revenue', 'Metascore']]
y = movies_df['Genre_encoded']

# تقسیم داده و استانداردسازی
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# آموزش مدل KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# ارزیابی دقت
accuracy = knn.score(X_test_scaled, y_test)
print(f"\nKNN Accuracy: {accuracy:.2f}")

# پیش‌بینی برای فیلم جدید
new_movie = np.array([[25000, 8.5, 90]])  # مقادیر نمونه
new_movie_scaled = scaler.transform(new_movie)

predicted_genre = knn.predict(new_movie_scaled)
print("\nPrediction for new movie:")
print("Raw features:", new_movie[0])
print("Scaled features:", new_movie_scaled[0])
print("Predicted genre code:", predicted_genre[0])
print("Predicted genre:", label_encoder_genre.inverse_transform(predicted_genre)[0])