import pandas as pd
import matplotlib as plt
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Define the path to the folder containing the CSV files
csv_folder_path = r'C:\Users\HCS2022\Desktop\CICEVSE2024_Dataset\csv files'

# Find all CSV files in the specified folder (power consumption and kernel events)
csv_files = glob.glob(os.path.join(csv_folder_path, '*.csv'))

# Load the CSV files into a dataframe
dfs = [pd.read_csv(file, low_memory=False) for file in csv_files]

# assign individual dataframes to reduce dimensionality
kernel_events = dfs[0]
power_consumption = dfs[1]

# subset kernel events
kernel_events_reduced = kernel_events[
    [
        'instructions', 'cache-misses', 'exc_taken', 'cpu-migrations', 'dTLB-store-misses',
        'l1d_cache_wr', 'L1-icache-loads', 'l2d_cache_rd', 'mem_access_rd', 'mem_access_wr',
        'kmem_kfree', 'net_net_dev_xmit', 'qdisc_qdisc_dequeue', 'raw_syscalls_sys_enter',
        'irq_softirq_raise', 'sched_sched_migrate_task', 'sched_sched_switch', 'syscalls_sys_enter_close',
        'syscalls_sys_enter_read', 'syscalls_sys_enter_write'
    ]
]

power_consumption.drop('time', axis=1, inplace=True)

# subset of 2000 random rows for both files 
num_rows = 2000
kernel_events_reduced_subset = kernel_events_reduced.sample(n=num_rows, random_state=42)
power_consumption_subset = power_consumption.sample(n=num_rows, random_state=42)

# combine both files (shape = 2000 x 29)
combined_data = pd.concat([kernel_events_reduced_subset.reset_index(drop=True),
                           power_consumption_subset.reset_index(drop=True)], axis=1)

# seperate the features (X) and the target (y)
y = combined_data['Label']
X = combined_data.drop('Label', axis=1)

# seperate data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# initialize model
trees = RandomForestClassifier(n_estimators=100, random_state=42)

# train the model
trees.fit(X_train, y_train)

# make predictions on test data
y_pred = trees.predict(X_test)

# evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")

# the above model does not work because i have categorical data 