import pandas as pd
import pickle


from sklearn.metrics import accuracy_score, hamming_loss, f1_score

with open('rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('xgboost_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

with open('cart_model.pkl', 'rb') as file:
    cart_model = pickle.load(file)

with open('knn_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)

with open('mlp_model.pkl', 'rb') as file:
    mlp_model = pickle.load(file)


df = pd.read_csv('data_generalization_test_clean.csv')

X = df.iloc[:, :33]
y = df.iloc[:, 33:]

# make predictions
y_pred_rf = rf_model.predict(X)
y_pred_xgb = xgb_model.predict(X)
y_pred_cart = cart_model.predict(X)
y_pred_knn = knn_model.predict(X)
y_pred_mlp = mlp_model.predict(X)


# calculate accuracy
accuracy_rf = accuracy_score(y, y_pred_rf)
accuracy_xgb = accuracy_score(y, y_pred_xgb)
accuracy_cart = accuracy_score(y, y_pred_cart)
accuracy_knn = accuracy_score(y, y_pred_knn)
accuracy_mlp = accuracy_score(y, y_pred_mlp)

print('<---------------------Generalization test--------------------->')

print("Accuracy (RF):", accuracy_rf)
print("Accuracy (XGBoost):", accuracy_xgb)
print("Accuracy (CART):", accuracy_cart)
print("Accuracy (KNN):", accuracy_knn)
print("Accuracy (MLP):", accuracy_mlp)
print()
# calculate Hamming Loss
hamming_rf = hamming_loss(y, y_pred_rf)
hamming_xgb = hamming_loss(y, y_pred_xgb)
hamming_cart = hamming_loss(y, y_pred_cart)
hamming_knn = hamming_loss(y, y_pred_knn)
hamming_mlp = hamming_loss(y, y_pred_mlp)

print("Hamming Loss (RF):", hamming_rf)
print("Hamming Loss (XGBoost):", hamming_xgb)
print("Hamming Loss (CART):", hamming_cart)
print("Hamming Loss (KNN):", hamming_knn)
print("Hamming Loss (MLP):", hamming_mlp)
print()
# calculate micro F1 score
f1_micro_rf = f1_score(y, y_pred_rf, average='micro')
f1_micro_xgb = f1_score(y, y_pred_xgb, average='micro')
f1_micro_cart = f1_score(y, y_pred_cart, average='micro')
f1_micro_knn = f1_score(y, y_pred_knn, average='micro')
f1_micro_mlp = f1_score(y, y_pred_mlp, average='micro')
print("Micro F1 Score (RF):", f1_micro_rf)
print("Micro F1 Score (XGBoost):", f1_micro_xgb)
print("Micro F1 Score (CART):", f1_micro_cart)
print("Micro F1 Score (KNN):", f1_micro_knn)
print("Micro F1 Score (MLP):", f1_micro_mlp)

