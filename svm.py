import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define the path to the folder containing the CSV files
csv_folder_path = r'C:\Users\HCS2022\Desktop\CICEVSE2024_Dataset\csv files'

# Find all CSV files in the specified folder (power consumption and kernel events)
csv_files = glob.glob(os.path.join(csv_folder_path, '*.csv'))

# Load the CSV files into dataframes
dfs = [pd.read_csv(file, low_memory=False) for file in csv_files]

# Assign individual dataframes to reduce dimensionality
kernel_events = dfs[0]
power_consumption = dfs[1]

# Subset kernel events -> all contain numerical data
kernel_events_reduced = kernel_events[
    [
        'instructions', 'cache-misses', 'exc_taken', 'cpu-migrations', 'dTLB-store-misses',
        'l1d_cache_wr', 'L1-icache-loads', 'l2d_cache_rd', 'mem_access_rd', 'mem_access_wr',
        'kmem_kfree', 'net_net_dev_xmit', 'qdisc_qdisc_dequeue', 'raw_syscalls_sys_enter',
        'irq_softirq_raise', 'sched_sched_migrate_task', 'sched_sched_switch', 'syscalls_sys_enter_close',
        'syscalls_sys_enter_read', 'syscalls_sys_enter_write'
    ]
]

# Drop time -> use in a later model
power_consumption.drop('time', axis=1, inplace=True)

# Subset of 5000 random rows for both files
num_rows = 5000

kernel_events_reduced_subset = kernel_events_reduced.sample(n=num_rows, random_state=42)
power_consumption_subset = power_consumption.sample(n=num_rows, random_state=42)

# Combine both files
combined_data = pd.concat([kernel_events_reduced_subset.reset_index(drop=True),
                           power_consumption_subset.reset_index(drop=True)], axis=1)

# One-hot encode the categorical columns: 'state', 'attack', 'attack-group', 'Label'
categorical_columns = ['State', 'Attack', 'Attack-Group', 'Label', 'interface']
combined_data_encoded = pd.get_dummies(combined_data, columns=categorical_columns)

# Remove rows with missing values
combined_data_encoded = combined_data_encoded.dropna()

# Separate the features (X) and the target (y)
# Assuming 'Label' is the target column and was one-hot encoded, we'll combine them into a single target variable.
label_columns = [col for col in combined_data_encoded.columns if col.startswith('Label_')]
y = combined_data_encoded[label_columns].idxmax(axis=1)
y = y.str.replace('Label_', '')

# Encode the target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Drop label columns from features
X = combined_data_encoded.drop(label_columns, axis=1)

# Separate data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Standardize the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the SVM model
svm_model = SVC(kernel='linear', random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)  # 5-fold cross-validation

# Print the mean cross-validation score
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean()}")

# Train the model on the training set
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = svm_model.predict(X_test_scaled)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy on Test Set: {accuracy}")
