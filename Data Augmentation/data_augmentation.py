from sklearn.decomposition import PCA
import pandas as pd
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('data.csv')

# Remove rows where all column values are duplicated
data = data.drop_duplicates()

# Separate features and target
features = data.drop(columns=['Target'])
target = data['Target']

# Count the number of samples in each class
class_counts = target.value_counts()

# Define a sampling strategy dictionary to oversample each class
# This will ensure that the size of each class becomes 3 times that of the largest class in the original dataset.
sampling_strategy = {cls: 3 * class_counts.max() for cls in class_counts.index}

# Apply SMOTE with the defined sampling strategy
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
features_resampled, target_resampled = smote.fit_resample(features, target)

# Convert back to DataFrame for further processing
data_resampled = pd.DataFrame(features_resampled, columns=features.columns)
data_resampled['Target'] = target_resampled

# Save the new dataset to a CSV file
data_resampled.to_csv('augmented_data.csv', index=False)

# You can now proceed with the rest of your ML pipeline