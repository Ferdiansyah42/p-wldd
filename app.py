import os
import json
import queue
import threading
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, send_file, session
from generate_dataset import generate_sample_dataset
from ga_svm import (
    load_data,
    preprocess_data,
    split_and_scale,
    run_genetic_algorithm,
    train_svm,
    evaluate_model,
    save_pipeline,
    load_pipeline,
    predict_recipient
)
from reports import generate_excel_report, generate_pdf_report
from sklearn.feature_selection import mutual_info_classif

app = Flask(__name__)
app.secret_key = "bps_academic_dashboard_secret_key"

# Directory configuration
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

ACTIVE_DATA_PATH = os.path.join(DATA_DIR, "active_poverty.csv")
STATE_PATH = os.path.join(DATA_DIR, "state.json")
PIPELINE_PATH = os.path.join(DATA_DIR, "model_pipeline.pkl")

# Global thread-safe queue for training SSE logs
train_queue = queue.Queue()
is_training = False
training_lock = threading.Lock()

# ──────────────────────────────────────────────
# STATE MANAGEMENT HELPERS
# ──────────────────────────────────────────────

def get_default_state():
    return {
        "dataset_loaded": False,
        "dataset_filename": "",
        "preprocessed": False,
        "ga_trained": False,
        "svm_trained": False,
        "n_dup": 0,
        "n_features_all": 0,
        "n_features_ga": 0,
        "selected_features": [],
        "all_features": [],
        "accuracy_baseline": 0.0,
        "precision_baseline": 0.0,
        "recall_baseline": 0.0,
        "f1_baseline": 0.0,
        "accuracy_opt": 0.0,
        "precision_opt": 0.0,
        "recall_opt": 0.0,
        "f1_opt": 0.0,
        "cm_baseline": [],
        "cm_opt": [],
        "report_baseline": "",
        "report_opt": "",
        "ga_history": {
            "generation": [],
            "best_fitness": [],
            "avg_fitness": [],
            "n_features": []
        }
    }

def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return get_default_state()
    return get_default_state()

def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=4)

# ──────────────────────────────────────────────
# ROUTING - 9 PAGES OF DASHBOARD
# ──────────────────────────────────────────────

@app.route('/')
def index_dashboard():
    state = load_state()
    
    # Calculate stats if data is loaded
    stats = {
        "total_population": 0,
        "eligible_count": 0,
        "ineligible_count": 0,
        "accuracy": state.get("accuracy_opt", 0.0)
    }
    
    if state.get("dataset_loaded") and os.path.exists(ACTIVE_DATA_PATH):
        try:
            df = load_data(ACTIVE_DATA_PATH)
            stats["total_population"] = len(df)
            
            # Identify target column (default to last column if status kelayakan not explicitly in state)
            target = state.get("target_col", df.columns[-1])
            if target in df.columns:
                vc = df[target].value_counts()
                # Handle either Layak/Tidak Layak or numeric 0/1
                for key, val in vc.items():
                    key_str = str(key).lower()
                    if "tidak" in key_str or key_str == "0":
                        stats["ineligible_count"] += int(val)
                    else:
                        stats["eligible_count"] += int(val)
        except Exception as e:
            print("Error loading dashboard stats:", e)
            
    return render_template("dashboard.html", state=state, stats=stats)

@app.route('/dataset')
def page_dataset():
    state = load_state()
    data_preview = []
    headers = []
    
    if state.get("dataset_loaded") and os.path.exists(ACTIVE_DATA_PATH):
        try:
            df = load_data(ACTIVE_DATA_PATH)
            headers = df.columns.tolist()
            # Convert first 50 rows to list of lists for clean rendering
            data_preview = df.head(50).values.tolist()
        except Exception as e:
            print("Error reading dataset for preview:", e)
            
    return render_template("dataset.html", state=state, headers=headers, data_preview=data_preview)

@app.route('/preprocessing')
def page_preprocessing():
    state = load_state()
    return render_template("preprocessing.html", state=state)

@app.route('/ga-selection')
def page_ga_selection():
    state = load_state()
    features_status = []
    all_feats = state.get("all_features", [])
    selected_set = set(state.get("selected_features", []))
    
    for idx, f in enumerate(all_feats, 1):
        features_status.append({
            "no": idx,
            "name": f,
            "selected": f in selected_set
        })
        
    # Dynamic parameter calculation based on dataset rows and features
    recommended_params = {
        "pop_size": 30,
        "generations": 50,
        "crossover_rate": 0.8,
        "mutation_rate": 0.1,
        "tournament_k": 3,
        "cv_folds": 3,
        "early_stop": 10,
        "num_rows": 0,
        "num_features": 0,
        "is_custom": False
    }
    
    if os.path.exists(ACTIVE_DATA_PATH):
        try:
            df = load_data(ACTIVE_DATA_PATH)
            num_rows = len(df)
            target_col = df.columns[-1]
            ignore_cols = ["NIK", "Nama", target_col]
            features_list = [c for c in df.columns if c not in ignore_cols]
            num_features = len(features_list)
            
            if num_features > 0:
                recommended_params["num_rows"] = num_rows
                recommended_params["num_features"] = num_features
                recommended_params["is_custom"] = True
                
                # Heuristic calculations for speed and representation
                # Pop size: 2 * features, bounded [10, 30] for safety/speed
                recommended_params["pop_size"] = max(10, min(30, int(num_features * 2)))
                # Generations: 1.5 * features, bounded [5, 20]
                recommended_params["generations"] = max(5, min(20, int(num_features * 1.5)))
                
                # Folds: 2 for large, 5 for small
                if num_rows < 150:
                    recommended_params["cv_folds"] = 5
                elif num_rows > 800:
                    recommended_params["cv_folds"] = 2
                else:
                    recommended_params["cv_folds"] = 3
                    
                recommended_params["tournament_k"] = max(2, min(4, recommended_params["pop_size"] // 5))
                recommended_params["early_stop"] = max(3, recommended_params["generations"] // 3)
        except Exception as e:
            print("Error calculating recommended parameters:", e)
            
    return render_template("ga_selection.html", state=state, features_status=features_status, recommended_params=recommended_params)

@app.route('/svm-training')
def page_svm_training():
    state = load_state()
    return render_template("svm_training.html", state=state)

@app.route('/evaluation')
def page_evaluation():
    state = load_state()
    return render_template("evaluation.html", state=state)

@app.route('/visualization')
def page_visualization():
    state = load_state()
    return render_template("visualization.html", state=state)

@app.route('/recipients')
def page_recipients():
    state = load_state()
    return render_template("recipients.html", state=state)

@app.route('/reports')
def page_reports():
    state = load_state()
    return render_template("reports.html", state=state)

# ──────────────────────────────────────────────
# API ENDPOINTS
# ──────────────────────────────────────────────

@app.route('/api/load-sample', methods=['POST'])
def api_load_sample():
    try:
        generate_sample_dataset(ACTIVE_DATA_PATH, num_rows=1000)
        state = get_default_state()  # Reset pipeline state
        state["dataset_loaded"] = True
        state["dataset_filename"] = "sample_poverty.csv"
        save_state(state)
        
        # Clean old pipeline if exists
        if os.path.exists(PIPELINE_PATH):
            os.remove(PIPELINE_PATH)
            
        return jsonify({"success": True, "message": "Dataset sampel berhasil dimuat."})
    except Exception as e:
        return jsonify({"success": False, "message": f"Gagal membuat dataset sampel: {str(e)}"})

@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "Tidak ada file diunggah."})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "Nama file kosong."})
        
    if not file.filename.endswith('.csv'):
        return jsonify({"success": False, "message": "File harus berformat CSV."})
        
    try:
        file.save(ACTIVE_DATA_PATH)
        state = get_default_state()  # Reset pipeline state
        state["dataset_loaded"] = True
        state["dataset_filename"] = file.filename
        save_state(state)
        
        # Clean old pipeline if exists
        if os.path.exists(PIPELINE_PATH):
            os.remove(PIPELINE_PATH)
            
        return jsonify({"success": True, "message": f"File '{file.filename}' berhasil diunggah."})
    except Exception as e:
        return jsonify({"success": False, "message": f"Gagal menyimpan file: {str(e)}"})

@app.route('/api/run-preprocess', methods=['POST'])
def api_run_preprocess():
    state = load_state()
    if not state.get("dataset_loaded") or not os.path.exists(ACTIVE_DATA_PATH):
        return jsonify({"success": False, "message": "Dataset belum diunggah."})
        
    try:
        # Read parameters from POST request
        req_data = request.get_json() or {}
        clean_data = req_data.get("clean_data", True)
        scale_data = req_data.get("scale_data", True)
        split_data = req_data.get("split_data", True)
        
        # Load and verify
        df = load_data(ACTIVE_DATA_PATH)
        
        # Execute preprocessing (ignores NIK, Nama)
        df_clean, X, y, identities, encoders, cat_cols, n_dup = preprocess_data(df, clean_data=clean_data)
        
        # Update state
        state["preprocessed"] = True
        state["n_dup"] = n_dup if clean_data else 0
        state["all_features"] = X.columns.tolist()
        state["n_features_all"] = len(X.columns)
        state["clean_data"] = clean_data
        state["scale_data"] = scale_data
        state["split_data"] = split_data
        save_state(state)
        
        return jsonify({
            "success": True, 
            "message": "Preprocessing selesai secara dinamis.",
            "n_dup": n_dup if clean_data else 0,
            "n_features": len(X.columns)
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"Error Preprocessing: {str(e)}"})

# ──────────────────────────────────────────────
# THREADED MODEL TRAINING & SSE STREAMING
# ──────────────────────────────────────────────

def background_training_worker(svm_params, ga_params):
    global is_training
    try:
        # Load custom selections from state
        state = load_state()
        clean_data = state.get("clean_data", True)
        scale_data = state.get("scale_data", True)
        split_data = state.get("split_data", True)
        
        # 1. Load & Preprocess
        train_queue.put({"type": "log", "message": "Memuat dataset aktif..."})
        df = load_data(ACTIVE_DATA_PATH)
        
        train_queue.put({"type": "log", "message": "Melakukan Preprocessing & Encoding..."})
        df_clean, X, y, identities, label_encoders, cat_cols, n_dup = preprocess_data(df, clean_data=clean_data)
        
        if clean_data:
            train_queue.put({"type": "log", "message": f"Pembersihan data selesai (Duplikat dihapus: {n_dup})."})
        else:
            train_queue.put({"type": "log", "message": "Pembersihan data dilewati (sesuai pilihan user)."})
            
        if split_data:
            train_queue.put({"type": "log", "message": "Membagi data latih & uji (Split 80:20)..."})
        else:
            train_queue.put({"type": "log", "message": "Pemisahan data dilewati, menggunakan 100% data untuk latih & uji."})
            
        X_train, X_test, y_train, y_test, scaler = split_and_scale(
            X, y, test_size=0.2, split_data=split_data, scale_data=scale_data
        )
        
        if scale_data:
            train_queue.put({"type": "log", "message": "Standardisasi StandardScaler selesai."})
        else:
            train_queue.put({"type": "log", "message": "Standardisasi StandardScaler dilewati."})
        
        # Callback for GA progress
        def ga_callback(status):
            train_queue.put({"type": "progress", "data": status})
            
        train_queue.put({"type": "log", "message": "Menjalankan optimasi seleksi fitur dengan Genetic Algorithm..."})
        best_chrom, best_fitness, ga_history = run_genetic_algorithm(
            X_train, y_train, svm_params, ga_params, progress_callback=ga_callback
        )
        
        selected_indices = np.where(best_chrom == 1)[0]
        selected_features = X_train.columns[selected_indices].tolist()
        all_features = X_train.columns.tolist()
        
        train_queue.put({"type": "log", "message": f"GA selesai. Fitur terpilih: {len(selected_features)}/{len(all_features)}."})
        
        # 2. Train baseline model (all features)
        train_queue.put({"type": "log", "message": "Melatih model SVM Baseline (Semua Fitur)..."})
        model_baseline = train_svm(X_train, y_train, svm_params)
        metrics_baseline, report_baseline, cm_baseline = evaluate_model(model_baseline, X_test, y_test)
        
        # 3. Train optimized model (GA selected features)
        train_queue.put({"type": "log", "message": "Melatih model SVM + GA (Fitur Terpilih)..."})
        X_train_ga = X_train[selected_features]
        X_test_ga = X_test[selected_features]
        model_opt = train_svm(X_train_ga, y_train, svm_params)
        metrics_opt, report_opt, cm_opt = evaluate_model(model_opt, X_test_ga, y_test)
        
        # 4. Save state and pipeline
        train_queue.put({"type": "log", "message": "Menyimpan model & parameter pipeline..."})
        
        target_col = df.columns[-1]  # Assume last column
        save_pipeline(
            PIPELINE_PATH, model_baseline, model_opt, scaler,
            label_encoders, selected_features, target_col, all_features
        )
        
        state = load_state()
        state.update({
            "ga_trained": True,
            "svm_trained": True,
            "n_features_ga": len(selected_features),
            "selected_features": selected_features,
            "accuracy_baseline": metrics_baseline["Accuracy"],
            "precision_baseline": metrics_baseline["Precision"],
            "recall_baseline": metrics_baseline["Recall"],
            "f1_baseline": metrics_baseline["F1-Score"],
            "accuracy_opt": metrics_opt["Accuracy"],
            "precision_opt": metrics_opt["Precision"],
            "recall_opt": metrics_opt["Recall"],
            "f1_opt": metrics_opt["F1-Score"],
            "cm_baseline": cm_baseline,
            "cm_opt": cm_opt,
            "report_baseline": report_baseline,
            "report_opt": report_opt,
            "ga_history": ga_history,
            "target_col": target_col
        })
        save_state(state)
        
        train_queue.put({"type": "done", "message": "Siklus pelatihan model selesai dengan sukses!", "state": state})
        
    except Exception as e:
        train_queue.put({"type": "error", "message": f"Kegagalan Pelatihan: {str(e)}"})
    finally:
        with training_lock:
            is_training = False

@app.route('/api/train-stream')
def api_train_stream():
    global is_training
    
    # Check if data is preprocessed first
    state = load_state()
    if not state.get("preprocessed"):
        return jsonify({"success": False, "message": "Lakukan Preprocessing terlebih dahulu."})
        
    # Read SVM params from query strings
    svm_params = {
        "kernel": request.args.get("kernel", "rbf"),
        "C": float(request.args.get("C", 1.0)),
        "gamma": request.args.get("gamma", "scale")
    }
    
    # Read GA params
    ga_params = {
        "pop_size": int(request.args.get("pop_size", 30)),
        "generations": int(request.args.get("generations", 50)),
        "crossover_rate": float(request.args.get("crossover_rate", 0.8)),
        "mutation_rate": float(request.args.get("mutation_rate", 0.1)),
        "tournament_k": int(request.args.get("tournament_k", 3)),
        "cv_folds": int(request.args.get("cv_folds", 3)),
        "elitism": request.args.get("elitism", "true").lower() == "true",
        "early_stop_patience": int(request.args.get("early_stop", 10))
    }
    
    with training_lock:
        if is_training:
            return jsonify({"success": False, "message": "Pelatihan model sedang berjalan."})
        is_training = True
        
    # Clear queue
    while not train_queue.empty():
        try:
            train_queue.get_nowait()
        except queue.Empty:
            break
            
    # Start thread
    threading.Thread(
        target=background_training_worker,
        args=(svm_params, ga_params),
        daemon=True
    ).start()
    
    def sse_event_stream():
        while True:
            try:
                msg = train_queue.get(timeout=20)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["type"] in ["done", "error"]:
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                
    return Response(sse_event_stream(), mimetype='text/event-stream')

# ──────────────────────────────────────────────
# RECIPIENTS SEARCH API
# ──────────────────────────────────────────────

@app.route('/api/search')
def api_search():
    query = request.args.get('query', '').strip()
    if not query:
        return jsonify({"success": False, "message": "Masukkan NIK atau Nama untuk pencarian."})
        
    if not os.path.exists(ACTIVE_DATA_PATH):
        return jsonify({"success": False, "message": "Dataset belum diunggah."})
        
    try:
        df = load_data(ACTIVE_DATA_PATH)
        
        # Search by NIK (exact or starts with) or Nama (substring, case insensitive)
        query_lower = query.lower()
        
        # We need NIK and Nama columns to search
        nik_col = [c for c in df.columns if c.lower() == 'nik']
        nama_col = [c for c in df.columns if c.lower() == 'nama']
        
        if not nik_col or not nama_col:
            return jsonify({"success": False, "message": "Kolom NIK atau Nama tidak ditemukan pada dataset."})
            
        nik_col = nik_col[0]
        nama_col = nama_col[0]
        
        # Filter
        match_mask = (df[nik_col].astype(str) == query) | (df[nama_col].astype(str).str.lower().str.contains(query_lower))
        matches = df[match_mask]
        
        if matches.empty:
            return jsonify({"success": False, "message": f"Penduduk dengan NIK atau Nama '{query}' tidak ditemukan."})
            
        # Limit to 5 matches to avoid overwhelming
        results = []
        
        # Check if pipeline is trained
        has_pipeline = os.path.exists(PIPELINE_PATH)
        pipeline = None
        if has_pipeline:
            try:
                pipeline = load_pipeline(PIPELINE_PATH)
            except Exception as e:
                print("Error loading pipeline during search:", e)
                has_pipeline = False
                
        for idx, row in matches.head(5).iterrows():
            nik_val = str(row[nik_col])
            nama_val = str(row[nama_col])
            
            # Get actual label in dataset
            target_col = df.columns[-1]
            actual_label = str(row[target_col])
            
            # Predict dynamic metrics if pipeline exists
            if has_pipeline and pipeline:
                try:
                    pred_label, confidence = predict_recipient(pipeline, row)
                    conf_pct = f"{int(confidence * 100)}%"
                except Exception as e:
                    print("Error predicting row:", e)
                    pred_label = actual_label
                    conf_pct = "N/A"
            else:
                # Fallback to actual label in dataset and 100% confidence
                pred_label = actual_label
                conf_pct = "N/A"
                
            results.append({
                "nik": nik_val,
                "nama": nama_val,
                "actual_label": actual_label,
                "pred_label": pred_label,
                "confidence": conf_pct,
                "details": {col: str(val) for col, val in row.items() if col not in [nik_col, nama_col, target_col]}
            })
            
        return jsonify({"success": True, "results": results, "model_active": has_pipeline})
        
    except Exception as e:
        return jsonify({"success": False, "message": f"Error saat mencari: {str(e)}"})

# ──────────────────────────────────────────────
# EXPORT SUMMARIES (EXCEL, PDF)
# ──────────────────────────────────────────────

@app.route('/api/export-excel')
def api_export_excel():
    state = load_state()
    if not state.get("ga_trained"):
        return "Model belum dilatih. Lakukan pelatihan terlebih dahulu.", 400
        
    try:
        excel_stream = generate_excel_report(state, ACTIVE_DATA_PATH)
        return send_file(
            excel_stream,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name="Laporan_Akademis_Bansos.xlsx"
        )
    except Exception as e:
        return f"Gagal mengekspor Excel: {str(e)}", 500

@app.route('/api/export-pdf')
def api_export_pdf():
    state = load_state()
    if not state.get("ga_trained"):
        return "Model belum dilatih. Lakukan pelatihan terlebih dahulu.", 400
        
    try:
        pdf_stream = generate_pdf_report(state, ACTIVE_DATA_PATH)
        return send_file(
            pdf_stream,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="Laporan_Akademis_Bansos.pdf"
        )
    except Exception as e:
        return f"Gagal mengekspor PDF: {str(e)}", 500

# ──────────────────────────────────────────────
# ANALYTIC CHARTS API
# ──────────────────────────────────────────────

@app.route('/api/visualization-data')
def api_visualization_data():
    if not os.path.exists(ACTIVE_DATA_PATH):
        return jsonify({"success": False, "message": "Dataset belum diunggah."})
        
    try:
        df = load_data(ACTIVE_DATA_PATH)
        target_col = df.columns[-1]
        
        # 1. Poverty count per region (Kab/Kota or Provinsi)
        # Take top 10 regions by population
        region_col = "Kab/Kota" if "Kab/Kota" in df.columns else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
        top_regions = df[region_col].value_counts().head(10).index.tolist()
        
        region_data = []
        for reg in top_regions:
            reg_df = df[df[region_col] == reg]
            # Count Layak (1, 1.0, layak, ya, yes) / Tidak Layak
            is_positive = (
                reg_df[target_col].astype(str).str.lower().str.strip().isin(["1", "1.0", "layak", "ya", "yes"]) |
                (reg_df[target_col].astype(str).str.lower().str.contains("layak") & ~reg_df[target_col].astype(str).str.lower().str.contains("tidak"))
            )
            layak = int(is_positive.sum())
            tidak_layak = len(reg_df) - layak
            region_data.append({
                "region": reg,
                "layak": layak,
                "tidak_layak": tidak_layak
            })
            
        # 2. Income Distribution histogram (bins)
        # Assume income or expenditure is in IDR
        inc_col = "Pendapatan" if "Pendapatan" in df.columns else [
            c for c in df.columns 
            if "pendapatan" in c.lower() or "income" in c.lower() or "pengeluaran" in c.lower() or "expenditure" in c.lower()
        ]
        income_bins = []
        income_labels = ["< 1Jt", "1Jt - 2Jt", "2Jt - 3Jt", "3Jt - 4Jt", "4Jt - 5Jt", "5Jt+"]
        
        if inc_col:
            inc_col = inc_col[0] if isinstance(inc_col, list) else inc_col
            incomes = df[inc_col].dropna().astype(float).copy()
            
            # Check if values are in thousands (Ribu Rupiah)
            if "ribu" in inc_col.lower() or incomes.max() < 150000:
                incomes = incomes * 1000
                
            # Check if values are annual
            if "tahun" in inc_col.lower() or "annual" in inc_col.lower() or "year" in inc_col.lower():
                incomes = incomes / 12
                
            bin_counts = [
                int((incomes < 1000000).sum()),
                int(((incomes >= 1000000) & (incomes < 2000000)).sum()),
                int(((incomes >= 2000000) & (incomes < 3000000)).sum()),
                int(((incomes >= 3000000) & (incomes < 4000000)).sum()),
                int(((incomes >= 4000000) & (incomes < 5000000)).sum()),
                int((incomes >= 5000000).sum())
            ]
            income_bins = bin_counts
        else:
            income_bins = [0] * len(income_labels)
            
        # 3. Target Variable Pie Chart
        vc_target = df[target_col].value_counts()
        target_pie = {str(k): int(v) for k, v in vc_target.items()}
        
        # 4. Feature Importance (Mutual Info Score)
        # Preprocess temporary to compute MI score
        df_clean, X, y, _, _, _, _ = preprocess_data(df)
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        importance_data = []
        for col_name, score in zip(X.columns, mi_scores):
            importance_data.append({
                "feature": col_name,
                "importance": float(score)
            })
        # Sort by importance descending
        importance_data = sorted(importance_data, key=lambda x: x["importance"], reverse=True)
        
        return jsonify({
            "success": True,
            "region_data": region_data,
            "income_data": {
                "labels": income_labels,
                "values": income_bins
            },
            "target_pie": target_pie,
            "importance_data": importance_data
        })
        
    except Exception as e:
        return jsonify({"success": False, "message": f"Gagal memuat visualisasi data: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
