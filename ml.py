import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import os
# Read data
data = pd.read_csv('VeryLargeSubsetSumDataset.csv')

# Split the data
data['subset_exists'] = data['subset_exists'].astype('category')
train_data, test_data = train_test_split(data, test_size=0.33, random_state=1148, stratify=data['subset_exists'])

# Decision Tree Model
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(train_data.drop('subset_exists', axis=1), train_data['subset_exists'])
predictions_dt = dt_classifier.predict(test_data.drop('subset_exists', axis=1))

# XGBoost Model
xgb_classifier = XGBClassifier()
xgb_classifier.fit(train_data.drop('subset_exists', axis=1), train_data['subset_exists'])
predictions_xgb = xgb_classifier.predict(test_data.drop('subset_exists', axis=1))

# Random Forest Model
rf_classifier = RandomForestClassifier(n_estimators=1000, max_features=2)
rf_classifier.fit(train_data.drop('subset_exists', axis=1), train_data['subset_exists'])
predictions_rf = rf_classifier.predict(test_data.drop('subset_exists', axis=1))

# Evaluate the models
# Decision Tree
cm_dt = confusion_matrix(test_data['subset_exists'], predictions_dt)
acc_dt = accuracy_score(test_data['subset_exists'], predictions_dt)

# XGBoost
cm_xgb = confusion_matrix(test_data['subset_exists'], predictions_xgb)
acc_xgb = accuracy_score(test_data['subset_exists'], predictions_xgb)

# Random Forest
cm_rf = confusion_matrix(test_data['subset_exists'], predictions_rf)
acc_rf = accuracy_score(test_data['subset_exists'], predictions_rf)

# Create a directory if it doesn't exist
output_dir = 'model_graphs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to plot ROC Curve
def plot_roc_curve(fpr, tpr, roc_auc, model_name):
    plt.figure()
    lw = 2  # Line width
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f'ROC_{model_name}.png'))
    plt.close()

# Function to calculate and plot ROC curve for each model
def evaluate_model(model, test_features, test_labels, model_name):
    probabilities = model.predict_proba(test_features)[:,1]
    fpr, tpr, _ = roc_curve(test_labels, probabilities)
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, model_name)
    
    # Precision-Recall curve and Average Precision
    precision, recall, _ = precision_recall_curve(test_labels, probabilities)
    average_precision = average_precision_score(test_labels, probabilities)
    plt.figure()
    plt.step(recall, precision, where='post', label=f'Average Precision = {average_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve: {model_name}')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_dir, f'Precision_Recall_{model_name}.png'))
    plt.close()

# Evaluate models
evaluate_model(dt_classifier, test_data.drop('subset_exists', axis=1), test_data['subset_exists'], "Decision Tree")
evaluate_model(xgb_classifier, test_data.drop('subset_exists', axis=1), test_data['subset_exists'], "XGBoost")
evaluate_model(rf_classifier, test_data.drop('subset_exists', axis=1), test_data['subset_exists'], "Random Forest")

# Print results
print("Decision Tree Accuracy:", acc_dt)
print("XGBoost Accuracy:", acc_xgb)
print("Random Forest Accuracy:", acc_rf)
