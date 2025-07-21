#Use XGBoostClassifier
from xgboost import XGBClassifier
model = XGBClassifier(early_stopping_rounds=10)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
------------------------------------------------------------
import matplotlib.pyplot as plt
model.feature_importances_
plt.barh(X.columns, model.feature_importances_)
print("Importance of parametres")
------------------------------------------------------------
# Predict on the test set
prediction = model.predict(X_test)
# Calculate mean squared error
mse_classification = mean_squared_error(y_test, prediction)
print(f"XGBoost regression mean squared error: {mse_classification:.4f}")
# Round predictions to integer categories and calculate classification accuracy
prediction_rounded = np.round(prediction).astype(int)
accuracy_classification = accuracy_score(y_test, prediction_rounded)
#RESULT:
#XGBoost regression mean squared error: 0.2845
#XGBoost regression rounded classification accuracy: 0.9129
------------------------------------------------------------
print(df['Weather Type'].value_counts())
print(df['Weather Type'].value_counts(normalize=True) * 100)
------------------------------------------------------------
#ConfusionMatrix for each Classifiers
#DECISIONTREE
ConfusionMatrixDisplay.from_estimator(
    dtree, X_test, y_test, cmap=plt.cm.Blues
)
plt.show()
#GRIDSEARCH
ConfusionMatrixDisplay.from_estimator(
    grid_search, X_test, y_test, cmap=plt.cm.Blues
)
plt.show()
#RANDOMFOREST
ConfusionMatrixDisplay.from_estimator(
    rf_clf, X_test, y_test, cmap=plt.cm.Blues
)
plt.show()
#XGBCLASSIFIER
ConfusionMatrixDisplay.from_estimator(
    model, X_test, y_test, cmap=plt.cm.Blues
)
plt.show()
print("Weather: 0>cloudy, 1>rainy, 2>Snowy, 3>sunny ")
print(
    "Considering the confusion matrix for DecisionTree, GridSearch, RandomForest, and XGBClassifier, we can gain some meaningful insights on model's performances"
)
------------------------------------------------------------
Weather: 0>cloudy, 1>rainy, 2>Snowy, 3>sunny 
#Considering the confusion matrix for DecisionTree, GridSearch, RandomForest, and XGBClassifier, we can gain some meaningful insights on model's performances
------------------------------------------------------------
#Seeing the graph, the most frequent false classification amongst the weather was that one between 1 and 0, which is Rainy and Cloudy. This is evidently plausible as the boundary in between those is quite obscure (e.g., shares similar atmospheric pressure).
#Let's consider Rainy (value:1) to be Positive and Cloudy (value:0) to be Negative. 
------------------------------------------------------------
Model_Name 	False-Positive 	False-Negative 	Total False-Classification
Decision Tree 	38 	33 	71
Grid Search 	37 	35 	72
Random Forest 	34 	35 	69
XGBoost Classifier 	32 	36 	68
------------------------------------------------------------
#As seen above, although the difference is little, we can see that XGBoost Classifier getting the best result out of all classification models here. Considering the feature of XGBoost, that it will weigh the failed samples and learn from its 'mistakes', it can be said that XGBoost performed quite well with the obscure target variables.
------------------------------------------------------------
