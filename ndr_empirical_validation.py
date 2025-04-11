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
]

# Update the columns in the dataset to match the correct column names
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

# Step 10: Integrate NDR for explanation
def ndr_explanation(prediction, instance):
    explanation = []
    
    if prediction == 1:  # Good Credit
        explanation.append("The model predicts a **Good Credit** because, based on **banking laws**,")

        if instance['CheckingAcctStatus'] == 3:
            explanation.append(" - Checking account status: No checking account, which negatively impacts credit.")
        if instance['Duration'] > 12:
            explanation.append(" - Duration: The loan duration is more than 12 months, indicating a positive sign for credit.")
        if instance['CreditHistory'] == 1:
            explanation.append(" - Credit history: Existing credits have been paid, a good indicator for creditworthiness.")
        if instance['CreditAmount'] > 3000:
            explanation.append(f" - Credit amount: A larger credit amount of {instance['CreditAmount']} DM could be a negative indicator if the person has not proven creditworthiness.")
        if instance['Employment'] > 1:
		 explanation.append(" - Employment status: The person has been employed for more than 1 year, which is positive for creditworthiness.")
        if instance['Housing'] == 1:
            explanation.append(" - Housing: Living in a house/own property, which typically provides better financial access.")
    
    elif prediction == 0:  # Bad Credit
        explanation.append("The model predicts a **Bad Credit** because, based on **banking laws**,")

        if instance['CheckingAcctStatus'] == 3:
            explanation.append(" - Checking account status: No checking account, which negatively impacts credit.")
        if instance['CreditAmount'] > 5000:
            explanation.append(f" - Credit amount: A larger credit amount of {instance['CreditAmount']} DM could be a negative indicator if the person has not proven creditworthiness.")
        if instance['Purpose'] == 2:
            explanation.append(" - Purpose: A risky purpose for the credit could increase the chance of default.")
        if instance['PersonalStatus'] == 0:
            explanation.append(" - Personal status: Male, divorced, or separated, which may show financial instability.")

    return "\n".join(explanation)

# Step 11: Generate explanations for predictions
print("\nExplanations for the first few predictions:\n")
for i in range(5):
    prediction = y_pred[i]
    instance = X_test.iloc[i]
    explanation = ndr_explanation(prediction, instance)
    print(f"Prediction: {'Good Credit' if prediction == 1 else 'Bad Credit'}")
    print(f"Explanation:\n{explanation}\n")