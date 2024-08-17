import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, hamming_loss

# read data
df = pd.read_csv('data_merged.csv')
df = df.drop_duplicates()

# features and labels
X = df.iloc[:, :33]
y = df.iloc[:, 33:]

# split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define the objectives
def rf_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 150)
    max_depth = trial.suggest_categorical('max_depth', [None, 10, 20, 30])
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)

    rf_classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    rf_classifier.fit(X_train, Y_train)
    Y_pred = rf_classifier.predict(X_train)
    return f1_score(Y_train, Y_pred, average='micro')


def cart_objective(trial):
    max_depth = trial.suggest_int('max_depth', 1, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

    cart_classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    cart_classifier.fit(X_train, Y_train)
    Y_pred = cart_classifier.predict(X_train)
    return f1_score(Y_train, Y_pred, average='micro')


def knn_objective(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 3, 20)

    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(X_train, Y_train)
    Y_pred = knn_classifier.predict(X_train)
    return f1_score(Y_train, Y_pred, average='micro')


def mlp_objective(trial):
    hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50)])
    alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-2)

    mlp_classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, max_iter=500, random_state=42)
    mlp_classifier.fit(X_train, Y_train)
    Y_pred = mlp_classifier.predict(X_train)
    return f1_score(Y_train, Y_pred, average='micro')


def xgb_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 150)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)

    xgb_classifier = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, use_label_encoder=False,
                                   eval_metric='mlogloss')
    multi_target_xgb = MultiOutputClassifier(xgb_classifier)
    multi_target_xgb.fit(X_train, Y_train)
    Y_pred = multi_target_xgb.predict(X_train)
    return f1_score(Y_train, Y_pred, average='micro')


# use optuna to find optimal fit
studies = {
    'Random Forest': optuna.create_study(direction='maximize'),
    'CART': optuna.create_study(direction='maximize'),
    'KNN': optuna.create_study(direction='maximize'),
    'MLP': optuna.create_study(direction='maximize'),
    'XGBoost': optuna.create_study(direction='maximize')
}

studies['Random Forest'].optimize(rf_objective, n_trials=50)
studies['CART'].optimize(cart_objective, n_trials=50)
studies['KNN'].optimize(knn_objective, n_trials=50)
studies['MLP'].optimize(mlp_objective, n_trials=50)
studies['XGBoost'].optimize(xgb_objective, n_trials=50)

# find the best parameters and retrain the models
best_rf_classifier = RandomForestClassifier(**studies['Random Forest'].best_params, random_state=42)
best_rf_classifier.fit(X_train, Y_train)

best_cart_classifier = DecisionTreeClassifier(**studies['CART'].best_params, random_state=42)
best_cart_classifier.fit(X_train, Y_train)

best_knn_classifier = KNeighborsClassifier(**studies['KNN'].best_params)
best_knn_classifier.fit(X_train, Y_train)

best_mlp_classifier = MLPClassifier(**studies['MLP'].best_params, max_iter=500, random_state=42)
best_mlp_classifier.fit(X_train, Y_train)

best_xgb_classifier = XGBClassifier(**studies['XGBoost'].best_params, use_label_encoder=False, eval_metric='mlogloss')
multi_target_xgb = MultiOutputClassifier(best_xgb_classifier)
multi_target_xgb.fit(X_train, Y_train)

# test models on the testing set
Y_pred_rf_test = best_rf_classifier.predict(X_test)
Y_pred_cart_test = best_cart_classifier.predict(X_test)
Y_pred_knn_test = best_knn_classifier.predict(X_test)
Y_pred_mlp_test = best_mlp_classifier.predict(X_test)
Y_pred_xgb_test = multi_target_xgb.predict(X_test)

# make predictions on the training set
Y_pred_rf_train = best_rf_classifier.predict(X_train)
Y_pred_cart_train = best_cart_classifier.predict(X_train)
Y_pred_knn_train = best_knn_classifier.predict(X_train)
Y_pred_mlp_train = best_mlp_classifier.predict(X_train)
Y_pred_xgb_train = multi_target_xgb.predict(X_train)

print('Performance on test set:')
print()
print('<-----------------------accuracy score-------------------------->')
# calculate accuracy score
accuracy_rf_test = accuracy_score(Y_test, Y_pred_rf_test)
accuracy_cart_test = accuracy_score(Y_test, Y_pred_cart_test)
accuracy_knn_test = accuracy_score(Y_test, Y_pred_knn_test)
accuracy_mlp_test = accuracy_score(Y_test, Y_pred_mlp_test)
accuracy_xgb_test = accuracy_score(Y_test, Y_pred_xgb_test)

print("RF accuracy (Test):", accuracy_rf_test)
print("XGBoost accuracy (Test):", accuracy_xgb_test)
print("CART accuracy (Test):", accuracy_cart_test)
print("KNN accuracy (Test):", accuracy_knn_test)
print("MLP accuracy (Test):", accuracy_mlp_test)

# calculate f1 score (micro)
print()
print('<-----------------------F1 score (micro)----------------------->')
f1_rf_test = f1_score(Y_test, Y_pred_rf_test, average='micro')
f1_cart_test = f1_score(Y_test, Y_pred_cart_test, average='micro')
f1_knn_test = f1_score(Y_test, Y_pred_knn_test, average='micro')
f1_mlp_test = f1_score(Y_test, Y_pred_mlp_test, average='micro')
f1_xgb_test = f1_score(Y_test, Y_pred_xgb_test, average='micro')

print("RF f1 score (micro, Test):", f1_rf_test)
print("XGBoost f1 score (micro, Test):", f1_xgb_test)
print("CART f1 score (micro, Test):", f1_cart_test)
print("KNN f1 score (micro, Test):", f1_knn_test)
print("MLP f1 score (micro, Test):", f1_mlp_test)

# calculate Hamming Distance (Test)
print()
print('<-----------------------Hamming Distance (Test)----------------------->')
hamming_rf_test = hamming_loss(Y_test, Y_pred_rf_test)
hamming_cart_test = hamming_loss(Y_test, Y_pred_cart_test)
hamming_knn_test = hamming_loss(Y_test, Y_pred_knn_test)
hamming_mlp_test = hamming_loss(Y_test, Y_pred_mlp_test)
hamming_xgb_test = hamming_loss(Y_test, Y_pred_xgb_test)

print("RF Hamming Distance (Test):", hamming_rf_test)
print("XGBoost Hamming Distance (Test):", hamming_xgb_test)
print("CART Hamming Distance (Test):", hamming_cart_test)
print("KNN Hamming Distance (Test):", hamming_knn_test)
print("MLP Hamming Distance (Test):", hamming_mlp_test)


print()
print('<-----------------------Training Set Performance----------------------->')


accuracy_rf_train = accuracy_score(Y_train, Y_pred_rf_train)
accuracy_cart_train = accuracy_score(Y_train, Y_pred_cart_train)
accuracy_knn_train = accuracy_score(Y_train, Y_pred_knn_train)
accuracy_mlp_train = accuracy_score(Y_train, Y_pred_mlp_train)
accuracy_xgb_train = accuracy_score(Y_train, Y_pred_xgb_train)


print("RF accuracy (Train):", accuracy_rf_train)
print("XGBoost accuracy (Train):", accuracy_xgb_train)
print("CART accuracy (Train):", accuracy_cart_train)
print("KNN accuracy (Train):", accuracy_knn_train)
print("MLP accuracy (Train):", accuracy_mlp_train)

print()
print('<-----------------------Training Set F1 score (micro)----------------------->')

f1_rf_train = f1_score(Y_train, Y_pred_rf_train, average='micro')
f1_cart_train = f1_score(Y_train, Y_pred_cart_train, average='micro')
f1_knn_train = f1_score(Y_train, Y_pred_knn_train, average='micro')
f1_mlp_train = f1_score(Y_train, Y_pred_mlp_train, average='micro')
f1_xgb_train = f1_score(Y_train, Y_pred_xgb_train, average='micro')


print("RF f1 score (micro, Train):", f1_rf_train)
print("XGBoost f1 score (micro, Train):", f1_xgb_train)
print("CART f1 score (micro, Train):", f1_cart_train)
print("KNN f1 score (micro, Train):", f1_knn_train)
print("MLP f1 score (micro, Train):", f1_mlp_train)


print()
print('<-----------------------Hamming Distance (Train)----------------------->')
hamming_rf_train = hamming_loss(Y_train, Y_pred_rf_train)
hamming_cart_train = hamming_loss(Y_train, Y_pred_cart_train)
hamming_knn_train = hamming_loss(Y_train, Y_pred_knn_train)
hamming_mlp_train = hamming_loss(Y_train, Y_pred_mlp_train)
hamming_xgb_train = hamming_loss(Y_train, Y_pred_xgb_train)

print("RF Hamming Distance (Train):", hamming_rf_train)
print("XGBoost Hamming Distance (Train):", hamming_xgb_train)
print("CART Hamming Distance (Train):", hamming_cart_train)
print("KNN Hamming Distance (Train):", hamming_knn_train)
print("MLP Hamming Distance (Train):", hamming_mlp_train)
