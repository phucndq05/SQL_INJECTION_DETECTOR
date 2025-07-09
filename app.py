# SQL_INJECTION_DETECTOR/app.py
from flask import Flask, render_template, request, url_for
import joblib
import os

app = Flask(__name__)

MODEL_DIR = "model"
# QUAN TRỌNG: CHỌN DATASET MUỐN WEB APP SỬ DỤNG
DATASET_FOR_WEB = "dataset4_train_std" # <-- Ở ĐÂY LÀ DATASET4

loaded_models = {} 
available_model_choices_on_web = [] 
vectorizer = None 

try:
    vectorizer_filename = f"vectorizer_{DATASET_FOR_WEB}.pkl" # Ví dụ: vectorizer_dataset4_train_std.pkl
    vectorizer_path = os.path.join(MODEL_DIR, vectorizer_filename)
    vectorizer = joblib.load(vectorizer_path)
    print(f"Đã tải thành công vectorizer: {vectorizer_path}")
except FileNotFoundError:
    print(f"LỖI NGHIÊM TRỌNG: Không tìm thấy file vectorizer '{vectorizer_filename}' trong thư mục '{MODEL_DIR}'.")
    print("Hãy chắc chắn bạn đã chạy 'training.py' thành công và file này tồn tại.")
except Exception as e:
    print(f"Lỗi khi tải vectorizer '{vectorizer_filename}': {e}")

# Các loại model bạn muốn cung cấp trên web (phải khớp với tên trong training.py)
model_types_for_web = ["RandomForest", "NaiveBayes", "XGBoost"] 

if vectorizer: 
    for model_type in model_types_for_web:
        # Tên file model sẽ có dạng: RandomForest_dataset4_train_std.pkl, ...
        model_file_name = f"{model_type}_{DATASET_FOR_WEB}.pkl"
        model_path = os.path.join(MODEL_DIR, model_file_name)
        try:
            loaded_models[model_file_name] = joblib.load(model_path)
            available_model_choices_on_web.append(model_file_name) 
            print(f"Đã tải model: {model_file_name}")
        except FileNotFoundError:
            print(f"LƯU Ý: Không tìm thấy model '{model_file_name}' (train trên '{DATASET_FOR_WEB}') trong thư mục '{MODEL_DIR}'.")
        except Exception as e:
            print(f"Lỗi khi tải model '{model_file_name}': {e}")
else:
    print("Không thể tải models vì vectorizer chưa được tải.")

if not available_model_choices_on_web and vectorizer:
    print(f"CẢNH BÁO: Không có model nào được tải thành công cho dataset '{DATASET_FOR_WEB}'. Website có thể không hoạt động đúng.")


@app.route('/', methods=['GET', 'POST'])
def home():
    result_message = None
    message_type = None 
    current_model_choice_key = available_model_choices_on_web[0] if available_model_choices_on_web else None
    query_text_from_user = "" 

    if request.method == 'POST':
        query_text_from_user = request.form.get('query')
        selected_model_key_from_form = request.form.get('model_choice') 
        
        if selected_model_key_from_form in loaded_models:
            current_model_choice_key = selected_model_key_from_form
        elif available_model_choices_on_web: 
             current_model_choice_key = available_model_choices_on_web[0]

        if not query_text_from_user:
            result_message = "Vui lòng nhập câu truy vấn."
            message_type = "malicious" 
        elif not vectorizer:
            result_message = "Lỗi: Vectorizer chưa được tải. Vui lòng kiểm tra console của server."
            message_type = "malicious"
        elif not current_model_choice_key or current_model_choice_key not in loaded_models:
            result_message = f"Lỗi: Model được chọn không hợp lệ hoặc chưa được tải."
            message_type = "malicious"
        else:
            try:
                active_model = loaded_models[current_model_choice_key]
                query_vectorized = vectorizer.transform([query_text_from_user]) 
                prediction = active_model.predict(query_vectorized)
                
                if prediction[0] == 1: 
                    result_message = "CÓ KHẢ NĂNG là câu truy vấn SQL độc hại (SQLi)."
                    message_type = "malicious" 
                else: 
                    result_message = "Đây là câu truy vấn BÌNH THƯỜNG."
                    message_type = "safe"      
            except Exception as e:
                result_message = f"Đã có lỗi xảy ra trong quá trình dự đoán: {e}"
                message_type = "malicious" 
                print(f"Lỗi dự đoán trên web: {e}")
                
    return render_template('index.html', 
                           result_message=result_message,
                           message_type=message_type, 
                           available_models=available_model_choices_on_web, 
                           current_model_choice=current_model_choice_key,    
                           query_text=query_text_from_user,                  
                           dataset_for_web_name=DATASET_FOR_WEB) # Truyền tên dataset đang dùng cho web để hiển thị          

if __name__ == '__main__':
    if not vectorizer or not loaded_models or not available_model_choices_on_web:
        print("CẢNH BÁO: Một hoặc nhiều thành phần (vectorizer/models) không tải được.")
        print("Hãy kiểm tra các thông báo lỗi ở trên và đảm bảo 'training.py' đã chạy thành công cho DATASET_FOR_WEB đã chọn.")
        print(f"DATASET_FOR_WEB hiện tại là: '{DATASET_FOR_WEB}'")
    app.run(debug=True, port=5000)