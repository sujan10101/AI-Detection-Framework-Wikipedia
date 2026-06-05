"""
Combined Text Classification Pipeline
Evaluates 4 different classifiers on binary classification task:
1. LIWC Features Classifier (Random Forest)
2. TF-IDF Classifier (LinearSVC)
3. Universal Sentence Encoder Classifier (Random Forest)
4. BERT/DistilBERT Classifier (Random Forest)

All classifiers use 5-fold cross-validation with SHAP analysis support for LIWC.
"""

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import warnings
import subprocess
import sys
import os

# NLTK for sentence tokenization
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_fpr(y_true, y_pred, pos_label=1):
    """False positive rate: FP / (FP + TN) for binary classification."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, pos_label]).ravel()
    negatives = fp + tn
    return fp / negatives if negatives > 0 else 0.0
from sklearn.feature_extraction.text import TfidfVectorizer

# SHAP for model interpretability (LIWC)
import shap
import matplotlib.pyplot as plt

# Transformers and PyTorch imports
from transformers import AutoTokenizer, AutoModel
import torch

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

# Set environment variables early (before any TensorFlow import attempt)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# Model Configuration for BERT/DistilBERT
USE_DISTILBERT = True  # Set to False for BERT, True for DistilBERT

if USE_DISTILBERT:
    BERT_MODEL_NAME = "distilbert-base-uncased"
    BERT_BATCH_SIZE = 32
    BERT_MODEL_LABEL = "DistilBERT"
else:
    BERT_MODEL_NAME = "bert-base-uncased"
    BERT_BATCH_SIZE = 16
    BERT_MODEL_LABEL = "BERT"


# ============================================================================
# CLASSIFIER 1: LIWC Features Classifier
# ============================================================================

class LIWCClassifier:
    """LIWC features classifier with 5-fold cross-validation and SHAP support."""
    
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.final_model = None  # Store final trained model for SHAP
        self.final_scaler = None  # Store final scaler for SHAP
        self.feature_names = None  # Store feature names for SHAP
        
    def evaluate_cv(self, X, y, n_splits=5):
        """Perform 5-fold cross-validation on LIWC features."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_accuracies = []
        fold_f1_scores = []
        fold_fpr_scores = []
        fold_times = []
        
        print(f"\n{'='*60}")
        print("LIWC Features - 5-Fold Cross-Validation")
        print(f"{'='*60}")
        print(f"Number of LIWC features: {X.shape[1]}")
        print(f"Number of samples: {X.shape[0]}")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nFold {fold}/{n_splits}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            start_time = time.time()
            
            # Scale features
            print("  Scaling features...", flush=True)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train classifier
            print("  Training Random Forest...", flush=True)
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train)
            
            # Predict and evaluate
            print("  Making predictions...", flush=True)
            y_pred = model.predict(X_val_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='binary')
            fpr = compute_fpr(y_val, y_pred)
            
            fold_time = time.time() - start_time
            fold_accuracies.append(accuracy)
            fold_f1_scores.append(f1)
            fold_fpr_scores.append(fpr)
            fold_times.append(fold_time)
            
            print(f"  Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f} | FPR: {fpr:.4f} | Time: {fold_time:.2f}s")
        
        return fold_accuracies, fold_f1_scores, fold_fpr_scores, fold_times
    
    def train_final_model(self, X, y, feature_names=None):
        """
        Train a final model on all data for SHAP analysis.
        
        Args:
            X: Array of LIWC features
            y: Array of labels
            feature_names: List of feature names
        
        Returns:
            Tuple of (X_scaled, y) where X_scaled is the transformed feature matrix
        """
        print("\n" + "="*60)
        print("Training Final Model for SHAP Analysis")
        print("="*60)
        
        # Store feature names
        self.feature_names = feature_names if feature_names is not None else np.array([f"feature_{i}" for i in range(X.shape[1])])
        
        # Scale features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        print("Training RandomForest classifier...")
        self.classifier.fit(X_scaled, y)
        
        # Store final components for SHAP
        self.final_model = self.classifier
        self.final_scaler = self.scaler
        
        print(f"Final model trained on {len(X)} samples with {X_scaled.shape[1]} features.")
        
        return X_scaled, y


def load_liwc_data(filepath):
    """Load and preprocess LIWC features dataset."""
    print(f"\nLoading LIWC dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"LIWC dataset shape: {df.shape}")
    
    # Metadata columns to exclude
    metadata_cols = ['pageid', 'article_id', 'title', 'content', 'categories',
                     'word_count', 'is_ai_flagged', 'Segment', 'WC']
    
    # Get LIWC feature columns (all columns except metadata)
    liwc_feature_cols = [col for col in df.columns if col not in metadata_cols]
    print(f"Number of LIWC features: {len(liwc_feature_cols)}")
    
    # Check for missing values
    missing_values = df[liwc_feature_cols].isna().sum().sum()
    print(f"Missing values in LIWC features: {missing_values}")
    
    # Handle missing values
    df = df.dropna(subset=liwc_feature_cols)
    
    # Check target variable
    print(f"\nTarget variable 'is_ai_flagged' distribution:")
    print(df['is_ai_flagged'].value_counts())
    print(f"Class balance: {df['is_ai_flagged'].value_counts(normalize=True)}")
    
    # Extract features and labels
    X = df[liwc_feature_cols].values.astype(np.float32)
    y = df['is_ai_flagged'].values
    
    print(f"\nFinal LIWC dataset size: {len(X)} samples")
    print(f"Number of features: {X.shape[1]}")
    
    return X, y, liwc_feature_cols


def generate_shap_explanations(classifier, X_sample, y_sample=None, max_samples=1000, output_dir="shap_results"):
    """
    Generate SHAP explanations for a trained classifier.
    
    Args:
        classifier: Trained classifier object (must have final_model, final_scaler, and feature_names attributes)
        X_sample: Sample data to explain (transformed/scaled features)
        y_sample: Optional labels for the sample data
        max_samples: Maximum number of samples to use for SHAP (for performance)
        output_dir: Directory to save SHAP visualizations and results
    
    Returns:
        shap_values: SHAP values array
        shap_explainer: SHAP explainer object
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating SHAP Explanations")
    print("="*60)
    
    # Check if classifier has required attributes
    if not hasattr(classifier, 'final_model') or classifier.final_model is None:
        raise ValueError("Classifier must have a trained final_model. Call train_final_model() first.")
    
    # Limit sample size for performance
    if len(X_sample) > max_samples:
        print(f"Limiting SHAP analysis to {max_samples} samples (from {len(X_sample)} total)")
        # Use stratified sampling if labels are available
        if y_sample is not None:
            X_sample, _, y_sample, _ = train_test_split(
                X_sample, y_sample, 
                train_size=max_samples, 
                stratify=y_sample, 
                random_state=42
            )
        else:
            indices = np.random.choice(len(X_sample), max_samples, replace=False)
            X_sample = X_sample[indices]
            if y_sample is not None:
                y_sample = y_sample[indices]
    
    # Verify we got the right number of samples
    actual_samples = len(X_sample)
    print(f"Using {actual_samples} samples for SHAP analysis...")
    
    # Create SHAP explainer (TreeExplainer for RandomForest)
    print("Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(classifier.final_model)
    
    # Calculate SHAP values
    print("Calculating SHAP values (this may take a while)...")
    shap_values = explainer.shap_values(X_sample)
    
    # Handle binary classification (SHAP returns list for binary)
    if isinstance(shap_values, list):
        # For binary classification, use the positive class (index 1)
        shap_values = shap_values[1]
    
    print(f"SHAP values calculated. Shape: {shap_values.shape}")
    
    # Save SHAP values to CSV
    if classifier.feature_names is not None:
        shap_df = pd.DataFrame(
            shap_values,
            columns=[f"SHAP_{name}" for name in classifier.feature_names]
        )
        shap_df.to_csv(os.path.join(output_dir, "shap_values.csv"), index=False)
        print(f"SHAP values saved to {output_dir}/shap_values.csv")
    
    # Generate and save visualizations
    print("\nGenerating SHAP visualizations...")
    
    # 1. Summary plot (bar plot of mean absolute SHAP values)
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, 
                         feature_names=classifier.feature_names,
                         show=False, plot_type="bar", max_display=30)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_summary_bar.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Summary bar plot saved to {output_dir}/shap_summary_bar.png")
    except Exception as e:
        print(f"  ✗ Error creating summary bar plot: {e}")
    
    # 2. Summary plot (beeswarm plot)
    try:
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample,
                         feature_names=classifier.feature_names,
                         show=False, max_display=30)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_summary_beeswarm.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Summary beeswarm plot saved to {output_dir}/shap_summary_beeswarm.png")
    except Exception as e:
        print(f"  ✗ Error creating summary beeswarm plot: {e}")
    
    # 3. Feature importance (mean absolute SHAP values)
    try:
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(feature_importance)[::-1][:30]
        
        if classifier.feature_names is not None:
            top_features = classifier.feature_names[top_indices]
            top_importance = feature_importance[top_indices]
            
            importance_df = pd.DataFrame({
                'Feature': top_features,
                'Mean_Abs_SHAP_Value': top_importance
            })
            importance_df.to_csv(os.path.join(output_dir, "shap_feature_importance.csv"), index=False)
            print(f"  ✓ Feature importance saved to {output_dir}/shap_feature_importance.csv")
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_importance)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Mean |SHAP Value|')
            plt.title('Top 30 LIWC Features by Mean Absolute SHAP Value')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "shap_feature_importance.png"), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Feature importance plot saved to {output_dir}/shap_feature_importance.png")
    except Exception as e:
        print(f"  ✗ Error creating feature importance: {e}")
    
    # 4. Waterfall plot for a few example instances
    try:
        n_examples = min(5, len(X_sample))
        # Get expected value (handle binary classification)
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
        else:
            base_value = explainer.expected_value
        
        for i in range(n_examples):
            # Create Explanation object for waterfall plot
            explanation = shap.Explanation(
                values=shap_values[i],
                base_values=base_value,
                data=X_sample[i],
                feature_names=classifier.feature_names
            )
            
            plt.figure(figsize=(12, 8))
            shap.waterfall_plot(explanation, show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_waterfall_example_{i+1}.png"), dpi=150, bbox_inches='tight')
            plt.close()
        print(f"  ✓ Waterfall plots saved for {n_examples} examples")
    except Exception as e:
        print(f"  ✗ Error creating waterfall plots: {e}")
    
    print(f"\nSHAP analysis complete! Results saved to {output_dir}/")
    
    return shap_values, explainer


# ============================================================================
# CLASSIFIER 2: TF-IDF Classifier
# ============================================================================

class TFIDFClassifier:
    """TF-IDF features classifier with 5-fold cross-validation."""
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Initialize TF-IDF classifier.
        
        Args:
            max_features: Maximum number of features to extract (default: 5000)
            ngram_range: Range of n-grams to use (default: (1, 2) for unigrams and bigrams)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            min_df=2,  # Ignore terms that appear in fewer than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
        self.classifier = LinearSVC(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        
    def evaluate_cv(self, X, y, n_splits=5):
        """Perform 5-fold cross-validation on TF-IDF features."""
        # Convert to numpy array if it's a list
        if isinstance(X, list):
            X = np.array(X, dtype=object)
        if isinstance(y, list):
            y = np.array(y)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_accuracies = []
        fold_f1_scores = []
        fold_fpr_scores = []
        fold_times = []
        
        print(f"\n{'='*60}")
        print("TF-IDF Features - 5-Fold Cross-Validation")
        print(f"{'='*60}")
        print(f"Max Features: {self.max_features}, N-gram Range: {self.ngram_range}")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nFold {fold}/{n_splits}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Extract TF-IDF features
            start_time = time.time()
            X_train_tfidf = self.vectorizer.fit_transform(X_train.tolist())
            X_val_tfidf = self.vectorizer.transform(X_val.tolist())
            
            # Convert sparse matrix to dense array for scaling
            X_train_tfidf = X_train_tfidf.toarray()
            X_val_tfidf = X_val_tfidf.toarray()
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train_tfidf)
            X_val_scaled = self.scaler.transform(X_val_tfidf)
            
            # Train classifier
            self.classifier.fit(X_train_scaled, y_train)
            
            # Predict and evaluate
            y_pred = self.classifier.predict(X_val_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='binary')
            fpr = compute_fpr(y_val, y_pred)
            
            fold_time = time.time() - start_time
            fold_accuracies.append(accuracy)
            fold_f1_scores.append(f1)
            fold_fpr_scores.append(fpr)
            fold_times.append(fold_time)
            
            print(f"  Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f} | FPR: {fpr:.4f} | Time: {fold_time:.2f}s")
        
        return fold_accuracies, fold_f1_scores, fold_fpr_scores, fold_times


def load_and_preprocess_data_full_text(filepath):
    """Load and preprocess the dataset without extracting first sentence (for TF-IDF)."""
    print(f"\nLoading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values in content column
    missing_content = df['content'].isna().sum()
    print(f"Missing values in 'content': {missing_content}")
    
    # Handle missing values
    df = df.dropna(subset=['content'])
    
    # Check target variable
    print(f"\nTarget variable 'is_ai_flagged' distribution:")
    print(df['is_ai_flagged'].value_counts())
    print(f"Class balance: {df['is_ai_flagged'].value_counts(normalize=True)}")
    
    # Prepare data
    X = df['content'].values
    y = df['is_ai_flagged'].values
    
    # Convert text to string if needed
    X = [str(text) for text in X]
    
    # Keep full text (no first sentence extraction for TF-IDF)
    print("\nUsing full text for TF-IDF (no sentence extraction)...")
    
    # Filter out empty texts
    valid_indices = [i for i, text in enumerate(X) if text.strip()]
    X = [X[i] for i in valid_indices]
    y = [y[i] for i in valid_indices]
    
    print(f"\nFinal dataset size: {len(X)} samples")
    print(f"Sample full text (first 100 chars): {X[0][:100]}..." if X else "No samples")
    
    return X, y


# ============================================================================
# CLASSIFIER 3: Universal Sentence Encoder (USE) Classifier
# ============================================================================

class USEClassifier:
    """Universal Sentence Encoder classifier with 5-fold cross-validation."""
    
    def __init__(self):
        self.model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.embedding_model = None
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        
    def load_model(self):
        """Load USE model from TensorFlow Hub."""
        # Lazy import TensorFlow - only import when actually needed
        # This avoids random_device errors on systems where TensorFlow has issues
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except Exception as e:
            raise ImportError(f"TensorFlow is not available. Cannot load USE model. Error: {e}")
        
        print("Loading Universal Sentence Encoder model...")

        # Force TensorFlow to use CPU only — RTX 5090 (sm_120 Blackwell) is not
        # yet supported by TensorFlow's JIT compiler, causing graph execution errors.
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            pass

        # Set TensorFlow to use single thread to avoid mutex issues
        try:
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
        except:
            pass  # Ignore if already configured
        
        # Set random seed (use numpy-based approach if tf.random fails)
        try:
            tf.random.set_seed(42)
        except Exception as e:
            # Fallback: use numpy seed if TensorFlow random fails
            print(f"Warning: Could not set TensorFlow random seed: {e}")
            print("Using numpy random seed instead.")
            np.random.seed(42)
        
        try:
            print("Downloading/loading model from TensorFlow Hub (this may take a few minutes)...")
            self.embedding_model = hub.load(self.model_url)
            print("USE model loaded successfully.")
        except Exception as e:
            print(f"Error loading USE model: {e}")
            raise
        
    def get_embeddings(self, texts, batch_size=32):
        """Extract embeddings for texts in batches."""
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting USE embeddings"):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model(batch).numpy()
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)
    
    def evaluate_cv(self, X, y, n_splits=5):
        """Perform 5-fold cross-validation."""
        # Convert to numpy array if it's a list
        if isinstance(X, list):
            X = np.array(X, dtype=object)
        if isinstance(y, list):
            y = np.array(y)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_accuracies = []
        fold_times = []
        fold_f1s = []
        fold_fprs = []
        
        print(f"\n{'='*60}")
        print("Universal Sentence Encoder - 5-Fold Cross-Validation")
        print(f"{'='*60}")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nFold {fold}/{n_splits}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Extract embeddings
            start_time = time.time()
            X_train_emb = self.get_embeddings(X_train.tolist())
            X_val_emb = self.get_embeddings(X_val.tolist())
            
            # Scale features
            X_train_emb_scaled = self.scaler.fit_transform(X_train_emb)
            X_val_emb_scaled = self.scaler.transform(X_val_emb)
            
            # Train classifier
            self.classifier.fit(X_train_emb_scaled, y_train)
            
            # Predict and evaluate
            y_pred = self.classifier.predict(X_val_emb_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average="binary")
            fpr = compute_fpr(y_val, y_pred)
            
            fold_time = time.time() - start_time
            fold_accuracies.append(accuracy)
            fold_times.append(fold_time)
            fold_f1s.append(f1)
            fold_fprs.append(fpr)
            
            print(f"  Accuracy: {accuracy:.4f} | F1: {f1:.4f} | FPR: {fpr:.4f} | Time: {fold_time:.2f}s")
        
        return fold_accuracies, fold_f1s, fold_fprs, fold_times


# ============================================================================
# CLASSIFIER 4: BERT/DistilBERT Classifier
# ============================================================================

class BERTClassifier:
    """BERT classifier with 5-fold cross-validation."""
    
    def __init__(self, model_name=None, max_length=512, batch_size=None):
        # Use default from config if not specified
        if model_name is None:
            model_name = BERT_MODEL_NAME
        if batch_size is None:
            batch_size = BERT_BATCH_SIZE
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        
    def load_model(self):
        """Load BERT/DistilBERT model and tokenizer."""
        print(f"Loading {BERT_MODEL_LABEL} model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"{BERT_MODEL_LABEL} model loaded successfully on {self.device}.")
        
    def get_embeddings(self, texts):
        """Extract BERT/DistilBERT embeddings using [CLS] token."""
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc=f"Extracting {BERT_MODEL_LABEL} embeddings"):
                batch_texts = texts[i:i+self.batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Get model output
                outputs = self.model(**encoded)
                
                # Use [CLS] token embedding (first token)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)
        
        return np.vstack(embeddings)
    
    def evaluate_cv(self, X, y, n_splits=5):
        """Perform 5-fold cross-validation."""
        # Convert to numpy array if it's a list
        if isinstance(X, list):
            X = np.array(X, dtype=object)
        if isinstance(y, list):
            y = np.array(y)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_accuracies = []
        fold_times = []
        fold_f1s = []
        fold_fprs = []
        
        print(f"\n{'='*60}")
        print(f"{BERT_MODEL_LABEL} - 5-Fold Cross-Validation")
        print(f"{'='*60}")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nFold {fold}/{n_splits}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Extract embeddings
            start_time = time.time()
            X_train_emb = self.get_embeddings(X_train.tolist())
            X_val_emb = self.get_embeddings(X_val.tolist())
            
            # Scale features
            X_train_emb_scaled = self.scaler.fit_transform(X_train_emb)
            X_val_emb_scaled = self.scaler.transform(X_val_emb)
            
            # Train classifier
            self.classifier.fit(X_train_emb_scaled, y_train)
            
            # Predict and evaluate
            y_pred = self.classifier.predict(X_val_emb_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average="binary")
            fpr = compute_fpr(y_val, y_pred)
            
            fold_time = time.time() - start_time
            fold_accuracies.append(accuracy)
            fold_times.append(fold_time)
            fold_f1s.append(f1)
            fold_fprs.append(fpr)
            
            print(f"  Accuracy: {accuracy:.4f} | F1: {f1:.4f} | FPR: {fpr:.4f} | Time: {fold_time:.2f}s")
        
        return fold_accuracies, fold_f1s, fold_fprs, fold_times


# ============================================================================
# Data Preprocessing Utilities
# ============================================================================

def get_first_n_sentences(text, n_sentences=10, max_chars=6000):
    """Extract the first N sentences from text (for USE + BERT)."""
    if not text or pd.isna(text):
        return ""

    text = str(text).strip()
    if not text:
        return ""

    # Sentence tokenize; fall back to char-slice if tokenization fails
    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = []

    if sentences:
        selected = " ".join(sentences[:n_sentences]).strip()
    else:
        selected = text

    # Keep input reasonably bounded (USE can take long text, but this keeps runtime stable)
    if len(selected) > max_chars:
        selected = selected[:max_chars].rsplit(" ", 1)[0].strip()

    return selected


# Backwards-compatible name used elsewhere in the script
def get_first_sentence(text):
    """Legacy alias: now returns the first ~10 sentences."""
    return get_first_n_sentences(text, n_sentences=10, max_chars=6000)


# Lazy-loaded tokenizer for token-based truncation (USE with 512-token limit)
_truncate_tokenizer = None


def get_truncate_tokenizer():
    """Get or create tokenizer for truncating text to N tokens (same as BERT/DistilBERT)."""
    global _truncate_tokenizer
    if _truncate_tokenizer is None:
        _truncate_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    return _truncate_tokenizer


def truncate_to_max_tokens(text, max_tokens=512):
    """Truncate text to at most max_tokens (WordPiece tokens, same definition as BERT)."""
    if not text or pd.isna(text):
        return ""
    text = str(text).strip()
    if not text:
        return ""
    tokenizer = get_truncate_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=False, max_length=max_tokens, truncation=True)
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()


def load_and_preprocess_data(filepath, text_mode="sentences", n_sentences=10, max_chars=6000, max_tokens=None):
    """Load and preprocess the dataset.

    text_mode:
      - "sentences": use first n_sentences capped by max_chars
      - "full": use the full text; if max_chars is None, no cap (whole text); else capped at max_chars
    max_tokens: if set (e.g. 512), truncate each text to this many tokens (WordPiece, same as BERT) after other preprocessing
    """
    print(f"\nLoading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values in content column
    missing_content = df['content'].isna().sum()
    print(f"Missing values in 'content': {missing_content}")
    
    # Handle missing values
    df = df.dropna(subset=['content'])
    
    # Check target variable
    print(f"\nTarget variable 'is_ai_flagged' distribution:")
    print(df['is_ai_flagged'].value_counts())
    print(f"Class balance: {df['is_ai_flagged'].value_counts(normalize=True)}")
    
    # Prepare data
    X = df['content'].values
    y = df['is_ai_flagged'].values
    
    # Convert text to string if needed
    X = [str(text) for text in X]
    
    if text_mode == "full":
        if max_chars is None:
            print("\nUsing full text (no character cap) for USE...")
            X = [str(text).strip() for text in tqdm(X, desc="Processing texts")]
        else:
            print(f"\nUsing full text (capped at {max_chars} chars)...")
            X = [
                str(text)[:max_chars].rsplit(" ", 1)[0].strip() if len(str(text)) > max_chars else str(text).strip()
                for text in tqdm(X, desc="Processing texts")
            ]
    else:
        # Sentence-windowed input (USE-friendly) while BERT/DistilBERT will still respect
        # its tokenizer max_length (512 tokens) via truncation.
        print(f"\nExtracting first {n_sentences} sentences from each text (cap {max_chars} chars)...")
        X = [get_first_n_sentences(text, n_sentences=n_sentences, max_chars=max_chars)
             for text in tqdm(X, desc="Processing texts")]

    if max_tokens is not None:
        print(f"\nTruncating each text to {max_tokens} tokens (WordPiece, for USE)...")
        X = [truncate_to_max_tokens(t, max_tokens=max_tokens) for t in tqdm(X, desc="Truncating to tokens")]

    # Filter out empty texts
    valid_indices = [i for i, text in enumerate(X) if text.strip()]
    X = [X[i] for i in valid_indices]
    y = [y[i] for i in valid_indices]
    
    print(f"\nFinal dataset size: {len(X)} samples")
    print(f"Sample text (first 100 chars): {X[0][:100]}..." if X else "No samples")
    
    return X, y


# ============================================================================
# Results Summary and Saving Functions
# ============================================================================

def print_results_summary(liwc_results=None, tfidf_results=None, use_results=None, bert_results=None):
    """Print formatted results summary for all classifiers."""
    print(f"\n{'='*80}")
    print("COMBINED EVALUATION RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    # LIWC Results
    if liwc_results is not None:
        liwc_accs, liwc_f1s, liwc_fprs, liwc_times = liwc_results
        print("1. LIWC Features — AI + Human Combined (Random Forest):")
        print(f"  Mean Accuracy: {np.mean(liwc_accs):.4f} ± {np.std(liwc_accs):.4f}")
        print(f"  Per-fold Accuracies: {[f'{acc:.4f}' for acc in liwc_accs]}")
        print(f"  Mean F1 Score: {np.mean(liwc_f1s):.4f} ± {np.std(liwc_f1s):.4f}")
        print(f"  Per-fold F1 Scores: {[f'{f1:.4f}' for f1 in liwc_f1s]}")
        print(f"  Mean FPR: {np.mean(liwc_fprs):.4f} ± {np.std(liwc_fprs):.4f}")
        print(f"  Per-fold FPR: {[f'{fpr:.4f}' for fpr in liwc_fprs]}")
        print(f"  Mean Time per Fold: {np.mean(liwc_times):.2f}s ± {np.std(liwc_times):.2f}s")
        print(f"  Total Time: {np.sum(liwc_times):.2f}s")
        print("\n" + "-"*80 + "\n")
    
    # TF-IDF Results
    if tfidf_results is not None:
        tfidf_accs, tfidf_f1s, tfidf_fprs, tfidf_times = tfidf_results
        print("2. TF-IDF Features (LinearSVC):")
        print(f"  Mean Accuracy: {np.mean(tfidf_accs):.4f} ± {np.std(tfidf_accs):.4f}")
        print(f"  Per-fold Accuracies: {[f'{acc:.4f}' for acc in tfidf_accs]}")
        print(f"  Mean F1 Score: {np.mean(tfidf_f1s):.4f} ± {np.std(tfidf_f1s):.4f}")
        print(f"  Per-fold F1 Scores: {[f'{f1:.4f}' for f1 in tfidf_f1s]}")
        print(f"  Mean FPR: {np.mean(tfidf_fprs):.4f} ± {np.std(tfidf_fprs):.4f}")
        print(f"  Per-fold FPR: {[f'{fpr:.4f}' for fpr in tfidf_fprs]}")
        print(f"  Mean Time per Fold: {np.mean(tfidf_times):.2f}s ± {np.std(tfidf_times):.2f}s")
        print(f"  Total Time: {np.sum(tfidf_times):.2f}s")
        print("\n" + "-"*80 + "\n")
    
    # USE Results
    if use_results is not None:
        use_accs, use_f1s, use_fprs, use_times = use_results
        print("3. Universal Sentence Encoder (Random Forest):")
        print(f"  Mean Accuracy: {np.mean(use_accs):.4f} ± {np.std(use_accs):.4f}")
        print(f"  Per-fold Accuracies: {[f'{acc:.4f}' for acc in use_accs]}")
        print(f"  Mean F1: {np.mean(use_f1s):.4f} ± {np.std(use_f1s):.4f}")
        print(f"  Per-fold F1: {[f'{f1:.4f}' for f1 in use_f1s]}")
        print(f"  Mean FPR: {np.mean(use_fprs):.4f} ± {np.std(use_fprs):.4f}")
        print(f"  Per-fold FPR: {[f'{fpr:.4f}' for fpr in use_fprs]}")
        print(f"  Mean Time per Fold: {np.mean(use_times):.2f}s ± {np.std(use_times):.2f}s")
        print(f"  Total Time: {np.sum(use_times):.2f}s")
        print("\n" + "-"*80 + "\n")
    
    # BERT/DistilBERT Results
    if bert_results is not None:
        bert_accs, bert_f1s, bert_fprs, bert_times = bert_results
        print(f"4. {BERT_MODEL_LABEL} (Random Forest):")
        print(f"  Mean Accuracy: {np.mean(bert_accs):.4f} ± {np.std(bert_accs):.4f}")
        print(f"  Per-fold Accuracies: {[f'{acc:.4f}' for acc in bert_accs]}")
        print(f"  Mean F1: {np.mean(bert_f1s):.4f} ± {np.std(bert_f1s):.4f}")
        print(f"  Per-fold F1: {[f'{f1:.4f}' for f1 in bert_f1s]}")
        print(f"  Mean FPR: {np.mean(bert_fprs):.4f} ± {np.std(bert_fprs):.4f}")
        print(f"  Per-fold FPR: {[f'{fpr:.4f}' for fpr in bert_fprs]}")
        print(f"  Mean Time per Fold: {np.mean(bert_times):.2f}s ± {np.std(bert_times):.2f}s")
        print(f"  Total Time: {np.sum(bert_times):.2f}s")
        print("\n" + "-"*80 + "\n")
    
    print(f"\n{'='*80}\n")


def save_results_to_csv(liwc_results=None, tfidf_results=None, use_results=None, bert_results=None,
                        output_file="combined_evaluation_results.csv"):
    """Save results to CSV file, merging new columns into existing file if present."""
    new_cols = {}

    if liwc_results is not None:
        liwc_accs, liwc_f1s, liwc_fprs, liwc_times = liwc_results
        new_cols['LIWC_Accuracy'] = list(liwc_accs) + [np.mean(liwc_accs), np.std(liwc_accs)]
        new_cols['LIWC_F1_Score'] = list(liwc_f1s) + [np.mean(liwc_f1s), np.std(liwc_f1s)]
        new_cols['LIWC_FPR']      = list(liwc_fprs) + [np.mean(liwc_fprs), np.std(liwc_fprs)]
        new_cols['LIWC_Time']     = list(liwc_times) + [np.mean(liwc_times), np.std(liwc_times)]

    if tfidf_results is not None:
        tfidf_accs, tfidf_f1s, tfidf_fprs, tfidf_times = tfidf_results
        new_cols['TFIDF_Accuracy'] = list(tfidf_accs) + [np.mean(tfidf_accs), np.std(tfidf_accs)]
        new_cols['TFIDF_F1_Score'] = list(tfidf_f1s) + [np.mean(tfidf_f1s), np.std(tfidf_f1s)]
        new_cols['TFIDF_FPR']      = list(tfidf_fprs) + [np.mean(tfidf_fprs), np.std(tfidf_fprs)]
        new_cols['TFIDF_Time']     = list(tfidf_times) + [np.mean(tfidf_times), np.std(tfidf_times)]

    if use_results is not None:
        use_accs, use_f1s, use_fprs, use_times = use_results
        new_cols['USE_Accuracy'] = list(use_accs) + [np.mean(use_accs), np.std(use_accs)]
        new_cols['USE_F1']       = list(use_f1s) + [np.mean(use_f1s), np.std(use_f1s)]
        new_cols['USE_FPR']      = list(use_fprs) + [np.mean(use_fprs), np.std(use_fprs)]
        new_cols['USE_Time']     = list(use_times) + [np.mean(use_times), np.std(use_times)]

    if bert_results is not None:
        bert_accs, bert_f1s, bert_fprs, bert_times = bert_results
        new_cols[f'{BERT_MODEL_LABEL}_Accuracy'] = list(bert_accs) + [np.mean(bert_accs), np.std(bert_accs)]
        new_cols[f'{BERT_MODEL_LABEL}_F1']       = list(bert_f1s) + [np.mean(bert_f1s), np.std(bert_f1s)]
        new_cols[f'{BERT_MODEL_LABEL}_FPR']      = list(bert_fprs) + [np.mean(bert_fprs), np.std(bert_fprs)]
        new_cols[f'{BERT_MODEL_LABEL}_Time']     = list(bert_times) + [np.mean(bert_times), np.std(bert_times)]

    # Load existing file and merge, or create from scratch
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        for col, values in new_cols.items():
            existing_df[col] = values
        results_df = existing_df
    else:
        fold_col = list(range(1, 6)) + ['Mean', 'Std']
        results_df = pd.DataFrame({'Fold': fold_col, **new_cols})

    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


# ============================================================================
# Main Execution Pipeline
# ============================================================================

def main():
    """
    Main evaluation pipeline for all 4 classifiers.
    
    Configuration flags:
    - RUN_LIWC: Evaluate LIWC Features Classifier
    - RUN_TFIDF: Evaluate TF-IDF Classifier
    - RUN_USE: Evaluate Universal Sentence Encoder Classifier
    - RUN_BERT: Evaluate BERT/DistilBERT Classifier
    - RUN_SHAP: Generate SHAP explanations for LIWC (if RUN_LIWC is True)
    """
    # Configuration flags - set to True to enable each classifier
    RUN_LIWC = True
    RUN_TFIDF = False
    RUN_USE = False
    RUN_BERT = False
    RUN_SHAP = False  # Generate SHAP explanations for LIWC
    
    # Dataset paths
    text_dataset_path = "dataset_balanced.csv"
    liwc_dataset_path = "LIWC-22 Results - dataset_balanced - LIWC Analysis.csv"
    
    # Initialize results storage
    liwc_results = None
    tfidf_results = None
    use_results = None
    bert_results = None
    
    print("\n" + "="*80)
    print("COMBINED CLASSIFIERS EVALUATION PIPELINE")
    print("="*80)
    print(f"\nActive Classifiers:")
    print(f"  1. LIWC Features: {'✓ Enabled' if RUN_LIWC else '✗ Disabled'}")
    print(f"  2. TF-IDF Features: {'✓ Enabled' if RUN_TFIDF else '✗ Disabled'}")
    print(f"  3. USE Embeddings: {'✓ Enabled' if RUN_USE else '✗ Disabled'}")
    print(f"  4. {BERT_MODEL_LABEL} Embeddings: {'✓ Enabled' if RUN_BERT else '✗ Disabled'}")
    print(f"  SHAP Analysis (LIWC): {'✓ Enabled' if RUN_SHAP and RUN_LIWC else '✗ Disabled'}")
    print("="*80 + "\n")
    
    # ========================================================================
    # CLASSIFIER 1: LIWC Features
    # ========================================================================
    if RUN_LIWC:
        if not os.path.exists(liwc_dataset_path):
            print(f"⚠️  Warning: LIWC dataset not found at {liwc_dataset_path}")
            print("Skipping LIWC evaluation.\n")
        else:
            try:
                print("\n" + "="*80)
                print("EVALUATING CLASSIFIER 1: LIWC FEATURES")
                print("="*80)
                
                X_liwc, y_liwc, feature_names = load_liwc_data(liwc_dataset_path)
                
                liwc_classifier = LIWCClassifier()
                liwc_results = liwc_classifier.evaluate_cv(X_liwc, y_liwc, n_splits=5)
                
                if RUN_SHAP:
                    try:
                        print("\n" + "="*80)
                        print("GENERATING SHAP EXPLANATIONS FOR LIWC")
                        print("="*80)
                        X_scaled, y_processed = liwc_classifier.train_final_model(
                            X_liwc, y_liwc, feature_names=feature_names)
                        generate_shap_explanations(
                            liwc_classifier,
                            X_scaled,
                            y_processed,
                            max_samples=1000,
                            output_dir="shap_results_liwc"
                        )
                        print("\n✓ SHAP analysis completed successfully!")
                    except Exception as e:
                        print(f"\n⚠️  Warning: SHAP analysis failed: {e}")
                
                print("\n✓ LIWC evaluation completed!")
                
            except Exception as e:
                print(f"\n✗ Error during LIWC evaluation: {e}")
                import traceback
                traceback.print_exc()
    
    # ========================================================================
    # CLASSIFIER 2: TF-IDF Features
    # ========================================================================
    if RUN_TFIDF:
        if not os.path.exists(text_dataset_path):
            print(f"⚠️  Warning: Text dataset not found at {text_dataset_path}")
            print("Skipping TF-IDF evaluation.\n")
        else:
            try:
                print("\n" + "="*80)
                print("EVALUATING CLASSIFIER 2: TF-IDF FEATURES")
                print("="*80)
                
                # Load and preprocess data for TF-IDF
                X_text, y_text = load_and_preprocess_data_full_text(text_dataset_path)
                
                # Initialize and evaluate TF-IDF classifier
                tfidf_classifier = TFIDFClassifier(max_features=5000, ngram_range=(1, 2))
                tfidf_results = tfidf_classifier.evaluate_cv(X_text, y_text, n_splits=5)
                
                print("\n✓ TF-IDF evaluation completed!")
                
            except Exception as e:
                print(f"\n✗ Error during TF-IDF evaluation: {e}")
                import traceback
                traceback.print_exc()
    
    # ========================================================================
    # CLASSIFIER 3: Universal Sentence Encoder (USE)
    # ========================================================================
    if RUN_USE:
        if not os.path.exists(text_dataset_path):
            print(f"⚠️  Warning: Text dataset not found at {text_dataset_path}")
            print("Skipping USE evaluation.\n")
        else:
            # Check if TensorFlow is available
            tensorflow_available = False
            print("\nChecking TensorFlow availability for USE model...")
            try:
                result = subprocess.run(
                    [sys.executable, '-c', 'import tensorflow as tf; import tensorflow_hub as hub; print("OK")'],
                    capture_output=True,
                    timeout=300,
                    text=True
                )
                if result.returncode == 0:
                    tensorflow_available = True
                    print("✓ TensorFlow is available. USE model can be used.")
                else:
                    print("✗ TensorFlow import failed. USE will be skipped.")
                    print(f"  Error: {result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                print("✗ TensorFlow import timed out. USE will be skipped.")
            except Exception as e:
                print(f"✗ Could not test TensorFlow: {e}. USE will be skipped.")
            
            if tensorflow_available:
                try:
                    print("\n" + "="*80)
                    print("EVALUATING CLASSIFIER 3: UNIVERSAL SENTENCE ENCODER")
                    print("="*80)
                    
                    # Load and preprocess data for USE
                    X_text, y_text = load_and_preprocess_data(
                        text_dataset_path,
                        text_mode="full",
                        max_chars=20000,
                        max_tokens=512
                    )
                    
                    # Initialize and load USE classifier
                    use_classifier = USEClassifier()
                    use_classifier.load_model()
                    
                    # Evaluate USE classifier
                    use_results = use_classifier.evaluate_cv(X_text, y_text, n_splits=5)
                    
                    print("\n✓ USE evaluation completed!")
                    
                except Exception as e:
                    print(f"\n✗ Error during USE evaluation: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("\n⚠️  Skipping USE evaluation (TensorFlow not available).")
    
    # ========================================================================
    # CLASSIFIER 4: BERT/DistilBERT
    # ========================================================================
    if RUN_BERT:
        if not os.path.exists(text_dataset_path):
            print(f"⚠️  Warning: Text dataset not found at {text_dataset_path}")
            print("Skipping {BERT_MODEL_LABEL} evaluation.\n")
        else:
            try:
                print("\n" + "="*80)
                print(f"EVALUATING CLASSIFIER 4: {BERT_MODEL_LABEL}")
                print("="*80)
                
                # Load and preprocess data for BERT/DistilBERT
                X_text, y_text = load_and_preprocess_data(
                    text_dataset_path,
                    text_mode="full",
                    max_chars=20000
                )
                
                # Initialize and load BERT classifier
                bert_classifier = BERTClassifier()
                bert_classifier.load_model()
                
                # Evaluate BERT classifier
                bert_results = bert_classifier.evaluate_cv(X_text, y_text, n_splits=5)
                
                print(f"\n✓ {BERT_MODEL_LABEL} evaluation completed!")
                
            except Exception as e:
                print(f"\n✗ Error during {BERT_MODEL_LABEL} evaluation: {e}")
                import traceback
                traceback.print_exc()
    
    # ========================================================================
    # Print and Save Combined Results
    # ========================================================================
    print("\n" + "="*80)
    print("FINALIZING RESULTS")
    print("="*80)
    
    # Print combined summary
    print_results_summary(
        liwc_results=liwc_results,
        tfidf_results=tfidf_results,
        use_results=use_results,
        bert_results=bert_results
    )
    
    # Save combined results to CSV
    save_results_to_csv(
        liwc_results=liwc_results,
        tfidf_results=tfidf_results,
        use_results=use_results,
        bert_results=bert_results,
        output_file="combined_evaluation_results.csv"
    )
    
    print("\n" + "="*80)
    print("✓ EVALUATION PIPELINE COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
