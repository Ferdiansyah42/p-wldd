import joblib
import pandas as pd
from generate_dataset import generate_sample_dataset
from ga_svm import load_data, preprocess_data, split_and_scale, train_svm

temp_csv = "data/test_temp_poverty.csv"
generate_sample_dataset(output_path=temp_csv, num_rows=50)

df_loaded = load_data(temp_csv)
print("Target column name:", df_loaded.columns[-1])
print("Target column type:", df_loaded.iloc[:, -1].dtype)

df_clean, X, y, _, encoders, _, _ = preprocess_data(df_loaded)
print("y unique values after preprocess:", y.unique())
print("y dtype after preprocess:", y.dtype)
print("Is '__target__' in encoders?", "__target__" in encoders)
if "__target__" in encoders:
    print("Encoder classes:", encoders["__target__"].classes_)

X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, test_size=0.2)
print("y_train unique values:", y_train.unique())

model = train_svm(X_train, y_train, {"kernel": "linear", "C": 1.0, "gamma": "scale"})
pred = model.predict(X_test.iloc[[0]])
print("Model prediction on first row:", pred)
print("Prediction type:", type(pred[0]))
