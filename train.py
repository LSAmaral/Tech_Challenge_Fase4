import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

os.makedirs("models", exist_ok=True)

df = pd.read_csv("Obesity.csv")

features_num = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
features_cat = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
target = 'Obesity'

X = df[features_num + features_cat]
y = df[target]

y_encoded = y.astype('category').cat.codes
label_mapping = dict(enumerate(y.astype('category').cat.categories))
joblib.dump(label_mapping, "models/label_mapping.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_cat)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

print("Treinando o modelo")
pipeline.fit(X_train, y_train)

acc = pipeline.score(X_test, y_test)
print(f"Nova Assertividade Real: {acc * 100:.2f}% (Meta: >75%)")


joblib.dump(pipeline, "models/obesity_pipeline.pkl")
print("Novo pipeline exportado com sucesso!")