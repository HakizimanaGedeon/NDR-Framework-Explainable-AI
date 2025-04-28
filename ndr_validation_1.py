import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load the dataset
data = pd.read_csv("german.data-numeric", delim_whitespace=True, header=None)

# Step 2: Inspect the dataset
print("Number of columns in dataset:", len(data.columns))

# Check the first few rows to understand the data
print(data.head())

# Step 3: Define column names based on dataset description
columns = [
    'CheckingAcctStatus', 'Duration', 'CreditHistory', 'Purpose', 'CreditAmount', 'Savings',
    'Employment', 'InstallmentRate', 'PersonalStatus', 'OtherDebtors', 'ResidenceSince',
    'Property', 'Age', 'OtherInstallmentPlans', 'Housing', 'ExistingCredits', 'Job',
    'NumDependents', 'Telephone', 'ForeignWorker', 'Class',  # 21 columns for data
    'AdditionalColumn1', 'AdditionalColumn2', 'AdditionalColumn3', 'AdditionalColumn4'  # 4 more columns
]  # Update the columns in the dataset to match the correct column names
if len(data.columns) == len(columns):
    data.columns = columns
else:
    print("Mismatch in number of columns!")
    print(f"Columns in the dataset: {data.columns.tolist()}")
    print(f"Expected columns: {columns}")
	# Step 4: Preprocess the data
# Encode categorical columns using LabelEncoder
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

encoder = LabelEncoder()
for column in categorical_columns:
    data[column] = encoder.fit_transform(data[column])

# Step 5: Split data into features (X) and target (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 7: Train a RandomForest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 8: Make predictions on the test set
y_pred = clf.predict(X_test)

# Step 9: Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

# Step 10: Feature Importance (for improving KB Loyalty)
feature_importance = clf.feature_importances_
feature_names = X.columns.tolist()

# Create a DataFrame of features and their importances
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("Feature Importance:")
print(feature_importance_df)

# Step 11: NDR Explanation
def ndr_explanation(prediction, instance):
    explanation = []
    
    if prediction == 1:  # Good Credit
        explanation.append("The model predicts a **Good Credit** because, based on **banking laws**,")
          # Check for important features based on their influence
        if instance['ForeignWorker'] == 1:
            explanation.append(" - Foreign worker status: Being a foreign worker is a positive factor for creditworthiness.")
        if instance['PersonalStatus'] == 1:
            explanation.append(" - Personal status: Stable family or marital status, which is positive for creditworthiness.")
        if instance['Duration'] > 12:
            explanation.append(" - Duration: The loan duration is more than 12 months, which indicates a positive sign for credit.")
        if instance['InstallmentRate'] < 3:
            explanation.append(" - Installment rate: A manageable installment rate, which indicates financial stability.")
    
    elif prediction == 0:  # Bad Credit
        explanation.append("The model predicts a **Bad Credit** because, based on **banking laws**,")
        
        # Check for features that indicate higher risk
        reasons = []
        
        # Check features with meaningful thresholds
        if instance['OtherDebtors'] > 1:  # High number of other debtors indicates higher risk
            reasons.append("Other debtors: The person has multiple financial obligations, which increases financial risk.")
        if instance['Purpose'] in [12, 13, 21, 23, 24]:  # Assuming purpose values like these indicate risky credit purposes
            reasons.append("Purpose: The credit purpose is considered risky, contributing to a negative credit prediction.")
        if instance['CreditAmount'] > 5000:  # High credit amount could be a strain
            reasons.append(f"Credit amount: A larger credit amount of {instance['CreditAmount']} DM could indicate financial strain.")
        if instance['Savings'] == 0:  # No savings, indicating financial instability
            reasons.append("Savings: Lack of savings, which can indicate financial instability.")
        if instance['Job'] == 0:  # Unstable job situation (assuming '0' means unemployed or unstable)
            reasons.append("Job: An unstable job situation, which contributes to higher financial risk.")
        if instance['Housing'] == 1:  # Housing status (renting) could indicate financial instability
            reasons.append("Housing: Renting a property could indicate financial instability compared to owning property.")
			  # Append reasons to explanation
        if reasons:
            explanation.extend(reasons)
        else:
            # If no specific reasons were found, fall back on a general statement
            explanation.append(" - The financial profile shows some risks that may not be immediately evident from the data, but multiple factors could contribute to a higher risk.")
    
    return "\n".join(explanation)

# Step 12: Evaluate explanations for a few predictions
def evaluate_kb_loyalty(instance, prediction):
    explanation = ndr_explanation(prediction, instance)
    
    # Check if the explanation follows the KB logic (monitor mismatch)
    if "Bad Credit" in explanation and prediction == 1:
        print(f"Mismatch detected! Predicted Good Credit, but KB suggests Bad Credit.")    
    return explanation

# Generate explanations for the first few predictions
print("\nExplanations for the first few predictions:\n")
for i in range(5):
    prediction = y_pred[i]
    instance = X_test.iloc[i]
    explanation = evaluate_kb_loyalty(instance, prediction)
    print(f"Prediction: {'Good Credit' if prediction == 1 else 'Bad Credit'}")
    print(f"Explanation:\n{explanation}\n")
	# Step 13: Calculate Rule Coverage and Mismatch Penalty (loyalty evaluation)
correct_predictions_kb = 0
mismatch_penalty = 0

for i in range(len(X_test)):
    prediction = y_pred[i]
    instance = X_test.iloc[i]
    explanation = evaluate_kb_loyalty(instance, prediction)
    
    # If explanation follows KB logic correctly, count it as a correct prediction
    if ("Good Credit" in explanation and prediction == 1) or ("Bad Credit" in explanation and prediction == 0):
        correct_predictions_kb += 1
    else:
        mismatch_penalty += 1

rule_coverage = correct_predictions_kb / len(X_test)
print(f"Correct Predictions based on KB logic: {correct_predictions_kb}")
print(f"Rule Coverage: {rule_coverage:.2f}")
print(f"Mismatch Penalty: {mismatch_penalty}")


