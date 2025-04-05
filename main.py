# Heart Disease Classification Project

# 1. Gerekli kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 2. Veri setini yükleme (heart.csv dosyası aynı klasörde olmalı)
df = pd.read_csv("heart.csv")
print("Veri seti boyutu:", df.shape)
print(df.head())

# 3. Eksik veri kontrolü
print("Eksik veriler:")
print(df.isnull().sum())

# 4. Hedef değişken: 'target'
X = df.drop("target", axis=1)
y = df["target"]

# 5. Veriyi eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Modelleri tanımlama
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC()
}

# 8. Modelleri eğitme ve değerlendirme
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Sonuçları:")
    print("Doğruluk Oranı:", acc)
    print(classification_report(y_test, y_pred))
    results.append((name, acc))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

# 9. Tüm sonuçları tablo olarak gösterme
results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
print("\nTüm Modellerin Doğruluk Karşılaştırması:")
print(results_df)

# 10. Sonuçları grafikleştirme
plt.figure(figsize=(10,6))
sns.barplot(data=results_df, x="Model", y="Accuracy")
plt.xticks(rotation=45)
plt.title("Modellere Göre Accuracy Karşılaştırması")
plt.ylim(0.7, 1.0)
plt.tight_layout()
plt.show()