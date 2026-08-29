# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.callbacks import EarlyStopping

# #  Load Data
# df = pd.read_csv("food_vendor_consultation.csv")

# #  Split Features & Target
# X = df.drop("vendor_category", axis=1)
# y = df["vendor_category"]

# # Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# # Feature Scaling 
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Build Neural Network

# model = Sequential([
#     Dense(3, activation='softmax', input_shape=(X_train.shape[1],))
# ])

# # Compile Model
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# # Early Stopping 
# early_stop = EarlyStopping(
#     monitor='val_loss',
#     patience=10,
#     restore_best_weights=True
# )

# # Train
# history = model.fit(
#     X_train, y_train,
#     validation_split=0.2,
#     epochs=100,
#     batch_size=20,
#     callbacks=[early_stop],
#     verbose=1
# )

# #  Evaluate
# loss, accuracy = model.evaluate(X_test, y_test)
# print("Test Accuracy:", accuracy)

# # Predictions
# y_pred_probs = model.predict(X_test)
# y_pred = np.argmax(y_pred_probs, axis=1)

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# Fix classification report issue by matching labels correctly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier

# Load dataset
df = pd.read_csv('geopolitical_dataset.csv')

target_col = df.columns[-1]

X = df.drop(columns=[target_col])
y = df[target_col]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# Train model
model = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.show()

# Classification Report (fix labels)
labels_present = np.unique(np.concatenate([y_test, y_pred]))
target_names = le.inverse_transform(labels_present)

report = classification_report(
    y_test, y_pred,
    labels=labels_present,
    target_names=target_names,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

# Plot classification report
plt.figure()
plt.imshow(report_df.iloc[:-1, :-1])
plt.title("Classification Report (Precision, Recall, F1-score)")
plt.xticks(range(len(report_df.columns[:-1])), report_df.columns[:-1], rotation=45)
plt.yticks(range(len(report_df.index[:-1])), report_df.index[:-1])
plt.colorbar()

for i in range(report_df.shape[0]-1):
    for j in range(report_df.shape[1]-1):
        plt.text(j, i, f"{report_df.iloc[i, j]:.2f}", ha='center', va='center')

plt.show()