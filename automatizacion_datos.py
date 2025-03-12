from pycaret.datasets import get_data

dataset = get_data("credit")
dataset.shape

data = dataset.sample(frac=0.95, random_state=786).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)
print("Datos para el modelo: " + str(data.shape))
print("Datos no visible para predicción: " + str(data_unseen.shape))

from pycaret.classification import *
exp_clf101 = setup(data = data, target = "default", session_id=10001)
best_model = compare_models()
print(best_model)

models()

best_model = compare_models()

dt = create_model("dt")
print(dt)

knn = create_model("knn")
print(knn)

rf = create_model("rf")
print(rf)

tuned_dt = tune_model(dt)
print(dt)
print(tuned_dt)

import numpy as np
tuned_knn = tune_model(knn, custom_grid = {"n_neighbors":np.arange(0, 50, 1)})
print(knn)
print(tuned_knn)
tuned_rf = tune_model(rf)
print(rf)
print(tuned_rf)

plot_model(tuned_rf, plot = "auc")
plot_model(tuned_rf, plot = "pr")
plot_model(tuned_rf, plot = "feature")
plot_model(tuned_rf, plot = "confusion_matrix")

evaluate_model(tuned_rf)
predict_model(tuned_rf)
final_rf = finalize_model(tuned_rf)
print(final_rf)
predict_model(final_rf)

unseen_predictions = predict_model(final_rf, data = data_unseen)
unseen_predictions.head()

from pycaret.utils.generic import check_metric
check_metric(unseen_predictions.default, unseen_predictions.prediction_label, "Accuracy")

save_model(final_rf, "Modelo final")
saved_final_ref = load_model("Modelo final")
new_prediction = predict_model(saved_final_ref, data = data_unseen)
new_prediction.head()