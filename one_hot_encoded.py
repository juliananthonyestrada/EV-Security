# I don't think this code works but keep just in case:

'''
# one-hot encode categorical data (Creates new columns and labels them as 1 or 0 for present or absent)
power_consumption_encoded = pd.get_dummies(power_consumption_subset, columns=[
    'State', 'Attack', 'Attack-Group', 'interface'
])

# replace old columns with new encoded ones
combined_data.drop(['State', 'Attack', 'Attack-Group', 'interface'], axis=1, inplace=True)
combined_data = pd.concat([combined_data, power_consumption_encoded], axis=1)

'''