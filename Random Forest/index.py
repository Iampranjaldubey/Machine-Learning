from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100,max_depth=3)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

print(predictions)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
xgb = XGBClassifier(n_estimators=100,max_depth=3,use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)

# Predict
xgb_predictions = xgb.predict(X_test)

print(predictions)