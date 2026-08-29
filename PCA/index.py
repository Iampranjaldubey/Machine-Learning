# X = df.drop(columns=["Malnutrition_Risk"])
# y = df[["Malnutrition_Risk"]]
# # Standardize the data
# scaler = StandardScaler();
# X_scaled = scaler.fit_transform(X)
# # reduce dimensionality to 15 components using iteration
# binary_columns = []
# for col in X.columns:
#     if X[col].nunique() == 2:
#         binary_columns.append(col)
# categorical_columns = []
# for col in X.columns:
#     if X[col].nunique() > 2 and X[col].nunique() < 20:
#         categorical_columns.append(col)
# continous_columns = []
# for col in X.columns:
#     if X[col].nunique() >= 20:
#         continous_columns.append(col)
# print("Binary columns: ", binary_columns)
# print("Categorical columns: ", categorical_columns)
# print("Continous columns: ", continous_columns)