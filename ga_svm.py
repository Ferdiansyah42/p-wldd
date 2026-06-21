import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def load_data(filepath):
    """Loads a CSV file, automatically detecting delimiter if needed."""
    try:
        # Try reading with comma
        df = pd.read_csv(filepath)
        if df.shape[1] <= 1:
            # Try reading with semicolon
            df = pd.read_csv(filepath, sep=";")
    except Exception as e:
        raise ValueError(f"Gagal membaca file CSV: {e}")
    
    if df.empty:
        raise ValueError("Dataset kosong.")
    return df

def preprocess_data(df, target_col=None, ignore_cols=["NIK", "Nama"], clean_data=True):
    """
    Cleans duplicates, handles missing values, encodes categories, and isolates NIK/Nama.
    Returns: df_clean, X, y, identities, label_encoders, cat_cols, n_dup
    """
    df_clean = df.copy()
    
    # Drop completely empty rows
    df_clean.dropna(how='all', inplace=True)
    
    # 1. Hapus duplikat
    n_dup = 0
    if clean_data:
        n_dup = int(df_clean.duplicated().sum())
        df_clean.drop_duplicates(inplace=True)
    
    # Identify target column
    if target_col is None or target_col not in df_clean.columns:
        target_col = df_clean.columns[-1]
        
    # Drop rows where target is NaN
    df_clean.dropna(subset=[target_col], inplace=True)
    
    y = df_clean[target_col].copy()
    
    # Isolate identities (NIK, Nama)
    actual_ignore = [col for col in ignore_cols if col in df_clean.columns]
    identities = df_clean[actual_ignore].copy()
    
    # Drop identities and target from training features
    X = df_clean.drop(columns=actual_ignore + [target_col])
    
    # Fill missing values
    num_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    cat_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    
    if clean_data:
        for col in num_cols:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())
                
        for col in cat_cols:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode()[0])
            
    # Label encode target if it's non-numeric
    target_encoder = None
    if not pd.api.types.is_numeric_dtype(y):
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y.astype(str)), index=y.index, name=target_col)
        
    # Label encoding for features
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        
    # Standardize target_encoder into label_encoders for convenience
    if target_encoder:
        label_encoders["__target__"] = target_encoder
        
    return df_clean, X, y, identities, label_encoders, cat_cols, n_dup

def split_and_scale(X, y, test_size=0.2, split_data=True, scale_data=True):
    """Splits data and/or standardizes features based on options."""
    if split_data:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X.copy(), X.copy(), y.copy(), y.copy()
        
    if scale_data:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )
    else:
        scaler = None
        X_train_scaled = X_train
        X_test_scaled = X_test
        
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ──────────────────────────────────────────────
# GENETIC ALGORITHM FOR FEATURE SELECTION
# ──────────────────────────────────────────────

def ga_fitness(individual, X_train, y_train, svm_params, cv_folds=3):
    selected = np.where(individual == 1)[0]
    if len(selected) == 0:
        return 0.0
        
    X_subset = X_train.iloc[:, selected]
    model = SVC(
        kernel=svm_params.get("kernel", "rbf"),
        C=svm_params.get("C", 1.0),
        gamma=svm_params.get("gamma", "scale"),
        random_state=42
    )
    
    min_class_size = y_train.value_counts().min()
    actual_folds = min(cv_folds, min_class_size)
    if actual_folds < 2:
        actual_folds = 2
        
    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_subset, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(scores.mean())

def run_genetic_algorithm(X_train, y_train, svm_params, ga_params, progress_callback=None):
    """
    Runs GA feature selection. 
    progress_callback is a function that takes a dict of stats at each generation.
    """
    n_features = X_train.shape[1]
    pop_size = ga_params.get("pop_size", 30)
    n_generations = ga_params.get("generations", 50)
    crossover_rate = ga_params.get("crossover_rate", 0.8)
    mutation_rate = ga_params.get("mutation_rate", 0.1)
    tournament_k = ga_params.get("tournament_k", 3)
    cv_folds = ga_params.get("cv_folds", 3)
    elitism = ga_params.get("elitism", True)
    early_stop = ga_params.get("early_stop_patience", 10)
    
    # Initialize Population
    population = np.random.randint(0, 2, size=(pop_size, n_features))
    for i in range(pop_size):
        if population[i].sum() == 0:
            population[i, np.random.randint(0, n_features)] = 1
            
    fitness_cache = {}
    history = {"generation": [], "best_fitness": [], "avg_fitness": [], "n_features": []}
    
    global_best_individual = None
    global_best_fitness = -1.0
    no_improve_count = 0
    gen_times = []
    
    for gen in range(n_generations):
        t_start = time.time()
        
        # Evaluate fitness
        fitness_scores = np.zeros(pop_size)
        cache_hits = 0
        
        for i in range(pop_size):
            chrom = population[i]
            key = tuple(chrom)
            if key in fitness_cache:
                fitness_scores[i] = fitness_cache[key]
                cache_hits += 1
            else:
                score = ga_fitness(chrom, X_train, y_train, svm_params, cv_folds)
                fitness_scores[i] = score
                fitness_cache[key] = score
                
        best_idx = np.argmax(fitness_scores)
        gen_best_fitness = fitness_scores[best_idx]
        gen_avg_fitness = fitness_scores.mean()
        gen_best_n_features = int(population[best_idx].sum())
        
        # Track global best
        if gen_best_fitness > global_best_fitness:
            global_best_fitness = gen_best_fitness
            global_best_individual = population[best_idx].copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Log history
        history["generation"].append(gen + 1)
        history["best_fitness"].append(global_best_fitness)
        history["avg_fitness"].append(gen_avg_fitness)
        history["n_features"].append(gen_best_n_features)
        
        elapsed = time.time() - t_start
        gen_times.append(elapsed)
        avg_time = np.mean(gen_times)
        eta = avg_time * (n_generations - (gen + 1))
        
        status = {
            "generation": gen + 1,
            "max_generations": n_generations,
            "best_fitness": global_best_fitness,
            "avg_fitness": gen_avg_fitness,
            "n_features": gen_best_n_features,
            "total_features": n_features,
            "cache_hits": cache_hits,
            "pop_size": pop_size,
            "elapsed": elapsed,
            "eta": eta,
            "early_stop_triggered": no_improve_count >= early_stop
        }
        
        if progress_callback:
            progress_callback(status)
            
        if no_improve_count >= early_stop:
            break
            
        # Recreate Population
        new_population = []
        if elitism:
            new_population.append(global_best_individual.copy())
            
        while len(new_population) < pop_size:
            # Tournament Selection
            indices1 = np.random.choice(pop_size, size=min(tournament_k, pop_size), replace=False)
            parent1 = population[indices1[np.argmax(fitness_scores[indices1])]].copy()
            
            indices2 = np.random.choice(pop_size, size=min(tournament_k, pop_size), replace=False)
            parent2 = population[indices2[np.argmax(fitness_scores[indices2])]].copy()
            
            # Crossover
            if np.random.rand() < crossover_rate:
                pt = np.random.randint(1, n_features)
                child1 = np.concatenate([parent1[:pt], parent2[pt:]])
                child2 = np.concatenate([parent2[:pt], parent1[pt:]])
            else:
                child1, child2 = parent1.copy(), parent2.copy()
                
            # Mutation
            mask1 = np.random.rand(n_features) < mutation_rate
            child1[mask1] = 1 - child1[mask1]
            if child1.sum() == 0:
                child1[np.random.randint(0, n_features)] = 1
                
            mask2 = np.random.rand(n_features) < mutation_rate
            child2[mask2] = 1 - child2[mask2]
            if child2.sum() == 0:
                child2[np.random.randint(0, n_features)] = 1
                
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
                
        population = np.array(new_population[:pop_size])
        
    return global_best_individual, global_best_fitness, history

# ──────────────────────────────────────────────
# MODEL TRAINING & EVALUATION
# ──────────────────────────────────────────────

def train_svm(X_train, y_train, svm_params):
    model = SVC(
        kernel=svm_params.get("kernel", "rbf"),
        C=svm_params.get("C", 1.0),
        gamma=svm_params.get("gamma", "scale"),
        probability=True,  # Set to True for confidence/probability scoring
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "F1-Score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    }
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()  # Convert to list for JSON serialization
    return metrics, report, cm

# ──────────────────────────────────────────────
# PREDICTION & PIPELINE SERIALIZATION
# ──────────────────────────────────────────────

def save_pipeline(filepath, model_baseline, model_opt, scaler, label_encoders, selected_features, target_col, features_list):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pipeline_state = {
        "model_baseline": model_baseline,
        "model_opt": model_opt,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "selected_features": selected_features,
        "target_col": target_col,
        "features_list": features_list
    }
    joblib.dump(pipeline_state, filepath)
    print(f"Pipeline saved to {filepath}")

def load_pipeline(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pipeline tidak ditemukan di {filepath}")
    return joblib.load(filepath)

def predict_recipient(pipeline_state, recipient_row):
    """
    Predicts eligibility for a single resident row (dict or pandas series).
    Returns class prediction and confidence probability.
    """
    model = pipeline_state["model_opt"]
    scaler = pipeline_state["scaler"]
    label_encoders = pipeline_state["label_encoders"]
    selected_features = pipeline_state["selected_features"]
    features_list = pipeline_state["features_list"]
    
    # Convert series to dict if necessary
    if isinstance(recipient_row, pd.Series):
        row_dict = recipient_row.to_dict()
    else:
        row_dict = dict(recipient_row)
        
    # Preprocess the row
    processed_row = {}
    for col in features_list:
        val = row_dict.get(col)
        
        # Missing values handling
        if val is None or pd.isna(val):
            # In simple terms, use a generic fallback
            val = "" if col in label_encoders else 0
            
        # Categorical encoding
        if col in label_encoders and col != "__target__":
            le = label_encoders[col]
            val_str = str(val)
            # Handle unseen categories safely
            if val_str in le.classes_:
                processed_row[col] = le.transform([val_str])[0]
            else:
                # Map to the first class or a default value
                processed_row[col] = 0
        else:
            try:
                processed_row[col] = float(val)
            except ValueError:
                processed_row[col] = 0.0
                
    # Create DataFrame with single row
    df_row = pd.DataFrame([processed_row], columns=features_list)
    
    # Scale features
    df_row_scaled = pd.DataFrame(scaler.transform(df_row), columns=features_list)
    
    # Filter features selected by GA
    df_row_ga = df_row_scaled[selected_features]
    
    # Predict probability and class
    prob = model.predict_proba(df_row_ga)[0]  # Array of probabilities [prob_0, prob_1]
    pred_class_encoded = int(model.predict(df_row_ga)[0])
    
    # Decode label
    if "__target__" in label_encoders:
        target_le = label_encoders["__target__"]
        pred_label = target_le.inverse_transform([pred_class_encoded])[0]
    else:
        pred_label = "Layak" if pred_class_encoded == 1 else "Tidak Layak"
        
    confidence = float(prob[pred_class_encoded])
    
    return pred_label, confidence
