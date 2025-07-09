# SQL_INJECTION_DETECTOR/training.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import json

# --- Cấu hình ---
DATA_DIR = "data"
MODEL_DIR = "model"

# Các dataset dùng để huấn luyện model (đã được chuẩn hóa bởi prepare_common_test_set.py)
# Tên file này phải khớp chính xác với tên file được tạo ra bởi prepare_common_test_set.py
TRAINING_DATASET_FILES = [ 
    "dataset1_train_std.csv",
    "dataset2_train_std.csv",
    "dataset3_train_std.csv",
    "dataset4_train_std.csv"
]
# Dataset dùng làm BỘ KIỂM THỬ CHUNG (đã được chuẩn hóa)
COMMON_TEST_DATASET_FILE = "datatest.csv" 
# Delimiter chung cho tất cả các file này sau khi đã được chuẩn hóa bởi prepare_common_test_set.py
UNIVERSAL_DELIMITER = ';' 

MODEL_TYPES_TO_TRAIN = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "NaiveBayes": MultinomialNB(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Tạo thư mục MODEL_DIR nếu nó chưa tồn tại
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Đã tạo thư mục: {MODEL_DIR}")

# Dictionary để lưu kết quả đánh giá của tất cả các model trên bộ test chung
evaluation_on_common_test_set = {}
# Dictionary để lưu trữ các vectorizer đã fit, key là tên ngắn gọn của dataset huấn luyện
fitted_vectorizers = {}

# --- Hàm tải và chuẩn bị dữ liệu (cho các file đã được chuẩn hóa) ---
def load_standardized_dataset(dataset_file_name):
    print(f"  Đang tải dataset chuẩn hóa: {dataset_file_name}...")
    dataset_path = os.path.join(DATA_DIR, dataset_file_name)
    try:
        # Tất cả các file đã được chuẩn hóa (bởi prepare_common_test_set.py) 
        # đều không có header và dùng UNIVERSAL_DELIMITER (là ';')
        df = pd.read_csv(dataset_path, delimiter=UNIVERSAL_DELIMITER, header=None, names=['query', 'label'], on_bad_lines='warn')
        df.dropna(subset=['query', 'label'], inplace=True) # Bỏ các dòng có giá trị rỗng ở cột query hoặc label
        if df.empty:
            print(f"  Lỗi: Dataset {dataset_file_name} rỗng sau khi loại bỏ NaN.")
            return None, None
        queries = df['query'].astype(str)
        labels = df['label'].astype(int)
        print(f"  Đã tải thành công {dataset_file_name}: {len(labels)} mẫu.")
        return queries, labels
    except FileNotFoundError:
        print(f"  Lỗi: Không tìm thấy file {dataset_path}.")
        return None, None
    except Exception as e:
        print(f"  Lỗi khi đọc file {dataset_path}: {e}")
        return None, None

# --- Bước 1: Huấn luyện Models trên từng Training Dataset ---
print("=== BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN MODEL ===")
for train_ds_filename in TRAINING_DATASET_FILES: 
    print(f"\n--- Huấn luyện trên dataset: {train_ds_filename} ---")
    train_queries, train_labels = load_standardized_dataset(train_ds_filename)
    
    if train_queries is None or train_labels is None:
        print(f"  Bỏ qua huấn luyện trên {train_ds_filename} do lỗi tải dữ liệu.")
        continue # Chuyển sang dataset huấn luyện tiếp theo

    # Lấy tên ngắn gọn của dataset huấn luyện để đặt tên cho model và vectorizer
    # Ví dụ: từ "dataset1_train_std.csv" sẽ lấy "dataset1_train_std"
    dataset_short_name_for_model = os.path.splitext(train_ds_filename)[0] 

    print(f"  Vector hóa dữ liệu cho {dataset_short_name_for_model}...")
    current_vectorizer = CountVectorizer(min_df=2, max_df=0.8)
    X_train_vectorized = current_vectorizer.fit_transform(train_queries) # Fit và transform trên dữ liệu huấn luyện
    
    vectorizer_path = os.path.join(MODEL_DIR, f"vectorizer_{dataset_short_name_for_model}.pkl")
    joblib.dump(current_vectorizer, vectorizer_path)
    fitted_vectorizers[dataset_short_name_for_model] = current_vectorizer # Lưu vectorizer đã fit để dùng sau
    print(f"  Đã lưu vectorizer: {vectorizer_path}")

    # Huấn luyện từng loại model (RandomForest, NaiveBayes)
    for model_type_name, model_prototype in MODEL_TYPES_TO_TRAIN.items():
        print(f"    Đang huấn luyện model: {model_type_name} trên {dataset_short_name_for_model}...")
        # Tạo một instance mới của model từ prototype để mỗi dataset huấn luyện có model riêng biệt
        model_instance = type(model_prototype)(**model_prototype.get_params())
        try:
            model_instance.fit(X_train_vectorized, train_labels) # Huấn luyện model
            
            # Tên file để lưu model, ví dụ: RandomForest_dataset1_train_std.pkl
            model_filename_for_save = f"{model_type_name}_{dataset_short_name_for_model}.pkl"
            model_path_to_save = os.path.join(MODEL_DIR, model_filename_for_save)
            joblib.dump(model_instance, model_path_to_save)
            print(f"      Đã lưu model: {model_path_to_save}")
        except Exception as e:
            print(f"      Lỗi khi huấn luyện model {model_type_name} trên {dataset_short_name_for_model}: {e}")

print("\n--- Hoàn tất quá trình huấn luyện model ---")


# --- Bước 2: Đánh giá tất cả các Model đã huấn luyện trên Bộ Kiểm thử Chung ---
print(f"\n=== BẮT ĐẦU ĐÁNH GIÁ TRÊN BỘ KIỂM THỬ CHUNG: {COMMON_TEST_DATASET_FILE} ===")
# Tải dữ liệu của bộ kiểm thử chung (datatest.csv)
common_test_queries, common_test_labels = load_standardized_dataset(COMMON_TEST_DATASET_FILE) 

if common_test_queries is not None and common_test_labels is not None:
    # Lặp qua từng dataset đã được dùng để huấn luyện
    for train_ds_filename_key in TRAINING_DATASET_FILES: 
        # Lấy tên ngắn gọn của dataset huấn luyện để tìm vectorizer và model tương ứng
        trained_on_dataset_short_name = os.path.splitext(train_ds_filename_key)[0] 
        
        vectorizer_to_use = fitted_vectorizers.get(trained_on_dataset_short_name)
        if not vectorizer_to_use:
            print(f"  Không tìm thấy vectorizer cho '{trained_on_dataset_short_name}'. Bỏ qua đánh giá model huấn luyện trên dataset này.")
            continue # Chuyển sang dataset huấn luyện tiếp theo

        print(f"\n  Vector hóa bộ kiểm thử chung bằng vectorizer của '{trained_on_dataset_short_name}'...")
        try:
            # Sử dụng vectorizer đã fit trên dataset huấn luyện để transform bộ test chung
            X_common_test_vectorized = vectorizer_to_use.transform(common_test_queries)
        except Exception as e:
            print(f"  Lỗi khi vector hóa bộ test chung bằng vectorizer của '{trained_on_dataset_short_name}': {e}. Bỏ qua.")
            continue # Chuyển sang dataset huấn luyện tiếp theo
            
        # Lặp qua từng loại model (RandomForest, NaiveBayes) đã được huấn luyện trên dataset này
        for model_type_name in MODEL_TYPES_TO_TRAIN.keys():
            # Xây dựng tên file model đã lưu
            model_filename_to_load = f"{model_type_name}_{trained_on_dataset_short_name}.pkl"
            model_path_to_load = os.path.join(MODEL_DIR, model_filename_to_load)
            
            # Tạo một định danh đầy đủ cho model để ghi vào báo cáo
            full_model_identifier_for_report = f"{model_type_name}_trained_on_{trained_on_dataset_short_name}" 

            try:
                print(f"    Đang tải model '{model_filename_to_load}' để đánh giá...")
                model_instance = joblib.load(model_path_to_load) # Tải model đã huấn luyện
                
                print(f"    Đánh giá '{full_model_identifier_for_report}' trên bộ test chung...")
                y_pred_common_test = model_instance.predict(X_common_test_vectorized) # Dự đoán trên bộ test chung

                # Tính toán các chỉ số
                accuracy = accuracy_score(common_test_labels, y_pred_common_test)
                precision = precision_score(common_test_labels, y_pred_common_test, zero_division=0)
                recall = recall_score(common_test_labels, y_pred_common_test, zero_division=0)
                f1 = f1_score(common_test_labels, y_pred_common_test, zero_division=0)
                
                # Lưu kết quả vào dictionary
                evaluation_on_common_test_set[full_model_identifier_for_report] = {
                    "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1
                }
                print(f"      Kết quả cho '{full_model_identifier_for_report}': Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

            except FileNotFoundError:
                print(f"    Lỗi: Không tìm thấy file model '{model_path_to_load}'. Model này có thể chưa được huấn luyện thành công.")
                evaluation_on_common_test_set[full_model_identifier_for_report] = {"Error": "Model file not found"}
            except Exception as e:
                print(f"    Lỗi khi tải hoặc đánh giá model '{model_filename_to_load}': {e}")
                evaluation_on_common_test_set[full_model_identifier_for_report] = {"Error": str(e)}
else:
    print("Lỗi: Không thể tải bộ kiểm thử chung. Dừng quá trình đánh giá.")

# Lưu kết quả đánh giá cuối cùng vào file JSON
results_json_path = os.path.join(MODEL_DIR, "evaluation_results_on_common_test.json") # Tên file JSON mới
try:
    with open(results_json_path, 'w') as f:
        json.dump(evaluation_on_common_test_set, f, indent=4) # indent=4 để file JSON dễ đọc
    print(f"\nĐã lưu kết quả đánh giá trên bộ test chung vào: {results_json_path}")
except Exception as e:
    print(f"Lỗi khi lưu file JSON chứa kết quả đánh giá: {e}")
    
print("\n--- Toàn bộ quá trình huấn luyện và đánh giá đã hoàn tất ---")