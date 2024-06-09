import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define the path to the folder containing the CSV files
csv_folder_path = r'C:\Users\HCS2022\Desktop\CICEVSE2024_Dataset\csv files'

# Find all CSV files in the specified folder (power consumption and kernel events)
csv_files = glob.glob(os.path.join(csv_folder_path, '*.csv'))

# Load the CSV files into a dataframe
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

# Subset of 500 random rows for both files
num_rows = 500

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
y = combined_data_encoded[[col for col in combined_data_encoded.columns if col.startswith('Label_')]]
X = combined_data_encoded.drop([col for col in combined_data_encoded.columns if col.startswith('Label_')], axis=1)

# Seperate data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Standardize the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize model
trees = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(trees, X, y, cv=5)  # 5-fold cross-validation

# Print the mean cross-validation score
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean()}")

# Train the model on the entire dataset
trees.fit(X, y)

# Make predictions on test data
y_pred = trees.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy on Test Set: {accuracy}")

# the model above works but it alwasy results in 100% accuracy and i cannot figure out why