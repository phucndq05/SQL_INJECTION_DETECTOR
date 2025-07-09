# SQL_INJECTION_DETECTOR/evaluation_reporter.py
import pandas as pd
import json
import os
from collections import defaultdict

MODEL_DIR = "model"
RESULTS_JSON_PATH = os.path.join(MODEL_DIR, "evaluation_results_on_common_test.json") 
REPORTS_DIR = "evaluation_reports"

if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
    print(f"Đã tạo thư mục: {REPORTS_DIR}")

def generate_formatted_evaluation_tables():
    try:
        with open(RESULTS_JSON_PATH, 'r') as f:
            results_data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file kết quả '{RESULTS_JSON_PATH}'.")
        print("Hãy chạy 'training.py' trước để tạo file này.")
        return None
    except json.JSONDecodeError:
        print(f"Lỗi: File '{RESULTS_JSON_PATH}' không phải là file JSON hợp lệ. Kiểm tra lại nội dung file.")
        return None
    except Exception as e:
        print(f"Lỗi không xác định khi đọc file JSON '{RESULTS_JSON_PATH}': {e}")
        return None

    if not results_data:
        print("Không có dữ liệu đánh giá trong file JSON để tạo bảng.")
        return None

    processed_data = defaultdict(lambda: defaultdict(dict))
    train_dataset_short_names_map = {} # Để map tên dài sang tên ngắn
    model_types = set()

    for full_model_identifier, metrics_dict in results_data.items():
        if "Error" in metrics_dict:
            print(f"Lưu ý: Model '{full_model_identifier}' có lỗi, sẽ được bỏ qua trong báo cáo.")
            continue
        
        try:
            parts = full_model_identifier.split("_trained_on_")
            model_type = parts[0]
            # train_dataset_full_name sẽ là "dataset1_train_std", "dataset2_train_std", v.v.
            train_dataset_full_name = parts[1] 
            
            # TẠO TÊN DATASET NGẮN GỌN CHO HÀNG CỦA BẢNG
            # Ví dụ: từ "dataset1_train_std" -> "dataset1"
            # Hoặc nếu tên gốc là "train_data_1_train_std" -> "train_data_1"
            # Chúng ta sẽ giả định tên file huấn luyện có dạng [tên_gốc]_train_std
            train_dataset_short_name = train_dataset_full_name.replace("_train_std", "")
            
            model_types.add(model_type)
            # Sử dụng tên ngắn gọn để làm key cho processed_data liên quan đến dataset
            train_dataset_short_names_map[train_dataset_full_name] = train_dataset_short_name 
            
            for metric_name, value in metrics_dict.items():
                processed_data[train_dataset_short_name][model_type][metric_name] = value
        except IndexError:
            print(f"Cảnh báo: Không thể phân tích định danh model '{full_model_identifier}'. Bỏ qua.")
            continue
        except Exception as e:
            print(f"Lỗi khi xử lý '{full_model_identifier}': {e}. Bỏ qua.")
            continue

    # Lấy danh sách tên dataset ngắn gọn đã được xử lý và sắp xếp
    unique_sorted_short_dataset_names = sorted(list(processed_data.keys()))

    if not unique_sorted_short_dataset_names or not model_types:
        print("Không có dữ liệu hợp lệ để tạo bảng sau khi xử lý.")
        return None

    sorted_model_types = sorted(list(model_types))
    
    metrics_to_tabulate = ["Accuracy", "Precision", "Recall", "F1-Score"]
    all_generated_tables_dfs = {}

    print("\nBắt đầu tạo bảng thống kê hiệu suất...\n")

    for metric in metrics_to_tabulate:
        data_for_current_metric_df = defaultdict(dict)
        
        for short_ds_name in unique_sorted_short_dataset_names: # Dùng tên dataset ngắn
            for model_type in sorted_model_types:
                value = processed_data.get(short_ds_name, {}).get(model_type, {}).get(metric, pd.NA)
                data_for_current_metric_df[model_type][short_ds_name] = value # Key trong là tên dataset ngắn
        
        if not data_for_current_metric_df:
            print(f"Không có dữ liệu để tạo bảng cho chỉ số: {metric}")
            continue

        df = pd.DataFrame(data_for_current_metric_df)
        # Đảm bảo các hàng (dataset ngắn) được sắp xếp
        df = df.reindex(unique_sorted_short_dataset_names) 
        df.index.name = "Dataset" # Đặt tên cho index (hàng) là "Dataset"
        
        all_generated_tables_dfs[metric] = df
        
        print(f"--- Bảng đánh giá cho chỉ số: {metric} ---")
        print(df.round(4).to_string()) 
        print("-" * 70) 

        csv_file_path = os.path.join(REPORTS_DIR, f"formatted_eval_table_{metric}.csv")
        try:
            df.round(4).to_csv(csv_file_path) 
            print(f"Đã lưu bảng '{metric}' vào: {csv_file_path}\n")
        except Exception as e:
            print(f"Lỗi khi lưu file CSV cho bảng '{metric}': {e}\n")

    return all_generated_tables_dfs

if __name__ == "__main__":
    print("--- Bắt đầu tạo bảng báo cáo đánh giá định dạng (Dòng: Dataset, Cột: Model) ---")
    generated_tables = generate_formatted_evaluation_tables()
    if generated_tables:
        print(f"--- Hoàn tất tạo các bảng báo cáo. Kiểm tra thư mục '{REPORTS_DIR}' ---")
    else:
        print("--- Không có bảng báo cáo nào được tạo (kiểm tra thông báo lỗi ở trên) ---")