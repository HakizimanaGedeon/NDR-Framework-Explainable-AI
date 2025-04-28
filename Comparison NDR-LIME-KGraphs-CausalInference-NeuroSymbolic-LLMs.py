import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import shap
from lime.lime_tabular import LimeTabularExplainer

# Step 1: Load the dataset
data = pd.read_csv("german.data-numeric", delim_whitespace=True, header=None)

# Step 2: Define column names
columns = [
    'CheckingAcctStatus', 'Duration', 'CreditHistory', 'Purpose', 'CreditAmount', 'Savings',
    'Employment', 'InstallmentRate', 'PersonalStatus', 'OtherDebtors', 'ResidenceSince',
    'Property', 'Age', 'OtherInstallmentPlans', 'Housing', 'ExistingCredits', 'Job',
    'NumDependents', 'Telephone', 'ForeignWorker', 'Class',
    'AdditionalColumn1', 'AdditionalColumn2', 'AdditionalColumn3', 'AdditionalColumn4'
]

if len(data.columns) == len(columns):
    data.columns = columns
else:
    print("Mismatch in number of columns!")

# Step 3: Encode categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
encoder = LabelEncoder()
for column in categorical_columns:
    data[column] = encoder.fit_transform(data[column])

# Step 4: Split data
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train RandomForest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Step 6: Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Step 7: Setup SHAP
explainer = shap.TreeExplainer(clf)
shap_values_all = explainer.shap_values(X_test)

# Step 7.5: Setup LIME
lime_explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['Bad Credit', 'Good Credit'],
    mode='classification'
)

# Step 8: Explanation Functions

def ndr_explanation(prediction, instance):
    explanation = []

    if prediction == 1:  # Good Credit
        explanation.append("The model predicts a **Good Credit** because, based on **banking laws**:")
        if instance['CheckingAcctStatus'] == 3:
            explanation.append(" - No checking account → negative credit impact.")
        if instance['Duration'] > 12:
            explanation.append(" - Loan duration >12 months → positive signal.")
        if instance['CreditHistory'] == 1:
            explanation.append(" - Existing credits paid → positive creditworthiness.")
        if instance['CreditAmount'] > 3000:
            explanation.append(f" - Large credit amount ({instance['CreditAmount']}) → may be risky.")
        if instance['Employment'] > 1:
            explanation.append(" - Employment >1 year → financially stable.")
        if instance['Housing'] == 1:
            explanation.append(" - Owns house/property → better access to credit.")
    else:  # Bad Credit
        explanation.append("The model predicts a **Bad Credit** because, based on **banking laws**:")
        if instance['CheckingAcctStatus'] == 3:
            explanation.append(" - No checking account → negative impact.")
        if instance['CreditAmount'] > 5000:
            explanation.append(f" - Credit amount ({instance['CreditAmount']}) is very high → risky.")
        if instance['Purpose'] == 2:
            explanation.append(" - Credit purpose is risky → higher default chance.")
        if instance['PersonalStatus'] == 0:
            explanation.append(" - Personal status shows possible financial instability.")
    return "\n".join(explanation)

def causal_explanation(instance):
    explanation = []
    if instance['Duration'] > 24:
        explanation.append("Long duration (>24 months) causally linked with credit risk.")
    if instance['Age'] < 25:
        explanation.append("Young age (<25) linked to higher default risk.")
    if instance['Employment'] > 3:
        explanation.append("Stable employment (>3 years) lowers risk.")
    return "Causal Inference Explanation:\n" + "\n".join(explanation) if explanation else "No strong causal factors detected."

def neuro_symbolic_explanation(instance):
    if instance['CreditAmount'] > 5000 and instance['Duration'] > 24:
        return "Neuro-Symbolic Rule: High credit amount and long duration → high risk."
    if instance['Employment'] > 2:
        return "Neuro-Symbolic Rule: Employment >2 years → lower risk."
    return "No strong symbolic rules triggered."

def knowledge_graph_explanation(instance):
    path = []
    if instance['CheckingAcctStatus'] == 3:
        path.append("No checking account → Financial instability")
    if instance['Savings'] < 2:
        path.append("Low savings → Poor financial backup")
    if instance['Housing'] == 1:
        path.append("Own property → Financial security")
    return "Knowledge Graph Path Explanation:\n" + " → ".join(path) if path else "No significant semantic path found."

def llm_explanation(instance):
    return (
        f"The individual has a credit amount of {instance['CreditAmount']} DM, "
        f"a duration of {instance['Duration']} months, and checking account status {instance['CheckingAcctStatus']}. "
        "These features collectively influence the credit decision made by the model."
    )

def lime_explanation(instance_index):
    explanation = lime_explainer.explain_instance(
        data_row=X_test.iloc[instance_index],
        predict_fn=clf.predict_proba,
        num_features=5
    )
    return explanation.as_list()

# Step 9: Run Explanation Comparison
print("\n--- Explanation Comparison for First Few Test Samples ---")
for i in range(3):  # Adjust the number of samples if needed
    instance_df = X_test.iloc[i:i+1]
    instance_row = X_test.iloc[i]
    prediction = y_pred[i]

    print(f"\n Sample #{i+1} — Prediction: {'Good Credit' if prediction == 1 else 'Bad Credit'}\n")

    print("▶ NDR Explanation:")
    print(ndr_explanation(prediction, instance_row))

    print("\n▶ Causal Inference Explanation:")
    print(causal_explanation(instance_row))

    print("\n▶ Neuro-Symbolic Explanation:")
    print(neuro_symbolic_explanation(instance_row))

    print("\n▶ Knowledge Graph Explanation:")
    print(knowledge_graph_explanation(instance_row))

    print("\n▶ LLM-style Explanation:")
    print(llm_explanation(instance_row))

    print("\n▶ LIME Explanation:")
    for feat, weight in lime_explanation(i):
        print(f" - {feat}: {weight:.4f}")

    print("\n" + "-" * 100)
