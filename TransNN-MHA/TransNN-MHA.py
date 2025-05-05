import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, classification_report, 
                           precision_score, recall_score, f1_score, 
                           confusion_matrix, roc_curve, auc, 
                           precision_recall_curve, average_precision_score)
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import time
import shap

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Data Loading and Preprocessing
print("Loading and preprocessing data...")
data = pd.read_csv('augmented_data.csv')
features = data.drop(columns=['Target'])
target = data['Target']

# Standardize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Encode labels
label_encoder = LabelEncoder()
target = label_encoder.fit_transform(target)
num_classes = len(np.unique(target))

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(features, target)

# Feature Selection
print("Performing feature selection...")
scores = np.zeros(X_resampled.shape[1])

# Mutual Information
mi = mutual_info_classif(X_resampled, y_resampled)
scores += mi

# ANOVA F-test
f_test, _ = f_classif(X_resampled, y_resampled)
scores += f_test

# Recursive Feature Elimination
model = LogisticRegression(max_iter=500)
rfe = RFE(model, n_features_to_select=30)
rfe.fit(X_resampled, y_resampled)
rfe_scores = np.array([1 if i in rfe.support_ else 0 for i in range(X_resampled.shape[1])])
scores += rfe_scores

# L1 Regularization
model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
model.fit(X_resampled, y_resampled)
lasso_scores = np.abs(model.coef_[0])
scores += lasso_scores

# Tree-based Importance
model = RandomForestClassifier()
model.fit(X_resampled, y_resampled)
tree_scores = model.feature_importances_
scores += tree_scores

# Average scores and select top features
average_scores = scores / 5
top_100_indices = np.argsort(average_scores)[-70:]
X_selected = features[:, top_100_indices]
X_selected_reshaped = X_selected.reshape(X_selected.shape[0], 1, X_selected.shape[1])

# 2. Enhanced Model Architecture with Attention Visualization
class TransformerModelWithAttention(tf.keras.Model):
    def __init__(self, input_shape, num_classes, num_heads=32, key_dim=64, ff_dim=512):
        super(TransformerModelWithAttention, self).__init__()
        self.input_layer = layers.InputLayer(input_shape=input_shape)
        self.encoder_layers = []
        
        for _ in range(6):
            self.encoder_layers.append({
                'attention': layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=0.2),
                'add': layers.Add(),
                'norm': layers.LayerNormalization(epsilon=1e-6),
                'ffn': layers.Dense(ff_dim, activation='relu')
            })
        
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(512, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.output_layer = layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs, return_attention=False):
        x = self.input_layer(inputs)
        attention_weights = []
        
        for layer in self.encoder_layers:
            attn_output, weights = layer['attention'](x, x, return_attention_scores=True)
            x = layer['add']([attn_output, x])
            x = layer['norm'](x)
            x = layer['ffn'](x)
            attention_weights.append(weights)
        
        x = self.global_pool(x)
        x = self.dense1(x)
        x = self.dropout(x)
        output = self.output_layer(x)
        
        if return_attention:
            return output, attention_weights
        return output

# 3. Cross-Validation with Enhanced Metrics
print("Starting cross-validation...")
n_splits = 6
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold = 1

# Initialize metrics storage
metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'roc_auc': [],
    'pr_auc': [],
    'confusion_matrices': [],
    'training_times': [],
    'inference_times': [],
    'histories': []  # Store training histories for plotting
}

# For attention visualization
attention_samples = []
sample_indices = []

for train_index, val_index in skf.split(X_selected_reshaped, target):
    print(f"\nFold {fold}/{n_splits}")
    
    X_train, X_val = X_selected_reshaped[train_index], X_selected_reshaped[val_index]
    y_train, y_val = target[train_index], target[val_index]
    
    # Convert to categorical for ROC/PR curves
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    
    # Build and compile model
    model = TransformerModelWithAttention(
        input_shape=(X_train.shape[1], X_train.shape[2]), 
        num_classes=num_classes
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
        
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    ]
    
    # Training with timing
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=110,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time
    metrics['training_times'].append(training_time)
    metrics['histories'].append(history.history)
    
    # Inference with timing
    start_time = time.time()
    y_pred_proba = model.predict(X_val)
    inference_time = (time.time() - start_time) / len(X_val)  # per sample
    metrics['inference_times'].append(inference_time)
    
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Store metrics
    metrics['accuracy'].append(accuracy_score(y_val, y_pred))
    metrics['precision'].append(precision_score(y_val, y_pred, average='weighted'))
    metrics['recall'].append(recall_score(y_val, y_pred, average='weighted'))
    metrics['f1'].append(f1_score(y_val, y_pred, average='weighted'))
    metrics['confusion_matrices'].append(confusion_matrix(y_val, y_pred))
    
    # ROC and PR curves
    fpr, tpr, _ = roc_curve(y_val_cat.ravel(), y_pred_proba.ravel())
    roc_auc = auc(fpr, tpr)
    metrics['roc_auc'].append(roc_auc)
    
    precision, recall, _ = precision_recall_curve(y_val_cat.ravel(), y_pred_proba.ravel())
    pr_auc = auc(recall, precision)
    metrics['pr_auc'].append(pr_auc)
    
    # Store attention weights for visualization (first sample from validation set)
    _, attention_weights = model(X_val[:1], return_attention=True)
    attention_samples.append(attention_weights)
    sample_indices.append(val_index[0])
    
    print(f"Fold {fold} Metrics:")
    print(f"Accuracy: {metrics['accuracy'][-1]:.4f}")
    print(f"Precision: {metrics['precision'][-1]:.4f}")
    print(f"Recall: {metrics['recall'][-1]:.4f}")
    print(f"F1-Score: {metrics['f1'][-1]:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Inference Time per Sample: {inference_time*1000:.2f}ms")
    
    fold += 1
