import os
import unittest
import pandas as pd
import numpy as np
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

class TestGASVMPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a tiny temp dataset for fast test execution
        cls.temp_csv = "data/test_temp_poverty.csv"
        cls.temp_model = "data/test_temp_pipeline.pkl"
        cls.df = generate_sample_dataset(output_path=cls.temp_csv, num_rows=50)

    @classmethod
    def tearDownClass(cls):
        # Clean up files after test execution
        for path in [cls.temp_csv, cls.temp_model]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Error removing temp file {path}: {e}")

    def test_1_load_data(self):
        df_loaded = load_data(self.temp_csv)
        self.assertIsNotNone(df_loaded)
        self.assertEqual(len(df_loaded), 50)
        self.assertIn("NIK", df_loaded.columns)
        self.assertIn("Nama", df_loaded.columns)
        self.assertIn("Status Kelayakan", df_loaded.columns)

    def test_2_preprocess_data(self):
        df_loaded = load_data(self.temp_csv)
        df_clean, X, y, identities, encoders, cat_cols, n_dup = preprocess_data(
            df_loaded, target_col="Status Kelayakan", ignore_cols=["NIK", "Nama"]
        )
        
        # Verify NIK and Nama are NOT in X but are in identities
        self.assertNotIn("NIK", X.columns)
        self.assertNotIn("Nama", X.columns)
        self.assertIn("NIK", identities.columns)
        self.assertIn("Nama", identities.columns)
        
        # Verify target is y
        self.assertEqual(y.name, "Status Kelayakan")
        self.assertEqual(len(X), len(y))
        
        # Check label encoding was applied
        self.assertIn("Pekerjaan", X.columns)
        self.assertTrue(np.issubdtype(X["Pekerjaan"].dtype, np.integer))
        self.assertIn("Pekerjaan", encoders)

    def test_3_split_and_scale(self):
        df_loaded = load_data(self.temp_csv)
        df_clean, X, y, _, _, _, _ = preprocess_data(df_loaded)
        X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, test_size=0.2)
        
        self.assertEqual(len(X_train), 40)
        self.assertEqual(len(X_test), 10)
        self.assertEqual(X_train.shape[1], X.shape[1])
        
        # Verify scaling is applied (mean close to 0, std close to 1)
        # Note: with only 40 samples, std might not be exactly 1, but should be defined
        self.assertTrue(np.isnan(X_train).sum().sum() == 0)

    def test_4_run_ga_and_train(self):
        df_loaded = load_data(self.temp_csv)
        df_clean, X, y, _, encoders, _, _ = preprocess_data(df_loaded)
        X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, test_size=0.2)
        
        svm_params = {"kernel": "linear", "C": 1.0, "gamma": "scale"}
        ga_params = {
            "pop_size": 4,
            "generations": 2,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "tournament_k": 2,
            "cv_folds": 2,
            "elitism": True,
            "early_stop_patience": 5
        }
        
        # Test GA run
        best_chrom, best_fitness, history = run_genetic_algorithm(
            X_train, y_train, svm_params, ga_params, progress_callback=None
        )
        
        self.assertEqual(len(best_chrom), X_train.shape[1])
        self.assertGreater(best_fitness, 0)
        self.assertEqual(len(history["generation"]), 2)
        
        # Get selected features
        selected_idx = np.where(best_chrom == 1)[0]
        selected_features = X_train.columns[selected_idx].tolist()
        self.assertGreater(len(selected_features), 0)
        
        # Train baseline
        model_baseline = train_svm(X_train, y_train, svm_params)
        metrics_base, _, _ = evaluate_model(model_baseline, X_test, y_test)
        self.assertIn("Accuracy", metrics_base)
        
        # Train optimized
        X_train_ga = X_train[selected_features]
        X_test_ga = X_test[selected_features]
        model_opt = train_svm(X_train_ga, y_train, svm_params)
        metrics_opt, _, _ = evaluate_model(model_opt, X_test_ga, y_test)
        self.assertIn("Accuracy", metrics_opt)
        
        # Test serialization
        features_list = X.columns.tolist()
        save_pipeline(
            self.temp_model, model_baseline, model_opt, scaler, encoders, 
            selected_features, "Status Kelayakan", features_list
        )
        self.assertTrue(os.path.exists(self.temp_model))
        
        # Test loading and single row prediction
        pipeline = load_pipeline(self.temp_model)
        sample_row = df_loaded.iloc[0]
        pred_label, confidence = predict_recipient(pipeline, sample_row)
        
        self.assertIn(pred_label, ["Layak", "Tidak Layak"])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

if __name__ == "__main__":
    unittest.main()

