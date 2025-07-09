# Hệ thống Phát hiện SQL Injection sử dụng Học máy

Dự án này xây dựng một hệ thống sử dụng các thuật toán học máy (Random Forest, Naive Bayes, và XGBoost) để phát hiện các câu truy vấn SQL có khả năng là tấn công SQL Injection (SQLi). Dự án bao gồm quy trình chuẩn bị dữ liệu từ các nguồn khác nhau, huấn luyện mô hình trên nhiều bộ dữ liệu, đánh giá trên một bộ kiểm thử chung, và một ứng dụng web demo (Flask) để kiểm tra chức năng.

**Thông tin:**
* **Trường:** Trường Đại học Công nghệ Thông tin - ĐHQG TP.HCM
* **Môn học:** Nhập môn bảo đảm và an ninh thông tin - IE105.P21
* **Giảng viên hướng dẫn:** TS. Nguyễn Tấn Cầm
* **Sinh viên thực hiện:**
    * Nguyễn Đặng Quang Phúc - 23521204
    * Trần Thị Như Phương - 23521249

---

## Quy trình Tổng quan

1.  **Chuẩn bị Dữ liệu Nguồn:** Cung cấp 4 file dataset gốc (ví dụ: `dataset1.csv`, `dataset2.csv`, `dataset3.csv`, `dataset4.csv`) vào thư mục `data/`. Các file này có thể có delimiter và header khác nhau.
2.  **Chuẩn hóa Dữ liệu và Tạo Bộ Test Chung (`prepare_common_test_set.py`):**
    * Script này đọc 4 file dataset nguồn, xử lý các định dạng (delimiter, header) khác nhau của chúng.
    * Trích 20% từ mỗi file nguồn để tạo thành một bộ kiểm thử chung duy nhất tên là `datatest.csv`.
    * 80% dữ liệu còn lại từ mỗi file nguồn sẽ được lưu thành các file riêng biệt dùng cho việc huấn luyện (ví dụ: `dataset1_train_std.csv`, `dataset2_train_std.csv`, v.v.).
    * **Quan trọng:** Tất cả các file được tạo ra ở bước này (`datatest.csv` và các file `*_train_std.csv`) đều được chuẩn hóa: **không có header** và sử dụng **dấu chấm phẩy (`;`)** làm delimiter.
3.  **Huấn luyện Model (`training.py`):**
    * Script này huấn luyện các model (Random Forest, Naive Bayes, XGBoost) trên từng file `datasetX_train_std.csv`.
    * Sau đó, tất cả các model đã huấn luyện sẽ được đánh giá trên cùng một bộ `datatest.csv`.
    * Kết quả đánh giá chi tiết được lưu vào `model/evaluation_results_on_common_test.json`.
    * Các model và vectorizer đã huấn luyện được lưu vào thư mục `model/`.
4.  **Xem Báo cáo Đánh giá (`evaluation_reporter.py`):**
    * Đọc file JSON kết quả và tạo các bảng so sánh hiệu suất (Accuracy, Precision, Recall, F1-Score).
    * Mỗi bảng có Dòng là "Dataset (Huấn luyện trên)" và Cột là "Model" (bao gồm cả XGBoost).
5.  **Chạy Ứng dụng Web Demo (`app.py`):**
    * Sử dụng model và vectorizer tốt nhất (dựa trên kết quả đánh giá) để người dùng có thể nhập câu truy vấn và nhận dự đoán.

---

## Cài đặt

1.  **Yêu cầu:**
    * Python 3.8+
    * pip (Trình quản lý gói Python)
    * Homebrew (cho macOS, để cài đặt `libomp` nếu gặp sự cố với XGBoost)

2.  **Tạo và kích hoạt môi trường ảo:**
    Mở Terminal trong thư mục gốc của dự án:
    ```bash
    python3 -m venv venv
    ```
    Kích hoạt môi trường ảo:
    * Trên macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    * Trên Windows:
        ```bash
        venv\Scripts\activate
        ```
    Bạn sẽ thấy `(venv)` ở đầu dòng lệnh.

3.  **Cài đặt `libomp` (chỉ cho macOS nếu gặp lỗi XGBoost):**
    ```bash
    brew install libomp
    ```

4.  **Cài đặt các thư viện Python cần thiết:**
    (Đảm bảo môi trường ảo đã được kích hoạt)
    ```bash
    pip3 install -r requirements.txt
    ```
    *(Lưu ý: File `requirements.txt` cần được tạo trước bằng lệnh `pip3 freeze > requirements.txt` sau khi đã cài đặt các thư viện cần thiết, bao gồm cả `xgboost`).*

---

## Cách chạy

1.  **Chuẩn bị Dữ liệu Nguồn:**
    * Đặt 4 file dataset gốc của bạn (ví dụ: `dataset1.csv`, `dataset2.csv`, `dataset3.csv`, `dataset4.csv` - đây là các file GỐC trước khi qua xử lý) vào thư mục `data/`.
    * **Quan trọng:** Mở file `prepare_common_test_set.py` và cập nhật dictionary `SOURCE_DATASET_FILES_CONFIG` cho đúng với tên file và đặc điểm (delimiter, header\_row, encoding ban đầu) của 4 file dataset gốc này. Xem hướng dẫn chi tiết trong code của `prepare_common_test_set.py`.

2.  **Chạy Script Chuẩn bị Dữ liệu (MỘT LẦN, hoặc khi dataset gốc thay đổi):**
    ```bash
    python3 prepare_common_test_set.py
    ```
    Thao tác này sẽ tạo ra các file `datasetX_train_std.csv` (dữ liệu huấn luyện đã chuẩn hóa) và `datatest.csv` (bộ test chung đã chuẩn hóa) trong thư mục `data/`.

3.  **Huấn luyện Model:**
    * (Tùy chọn) Xóa các file `.pkl` và `.json` cũ trong thư mục `model/` nếu muốn huấn luyện lại hoàn toàn.
    * Mở file `training.py` và đảm bảo danh sách `TRAINING_DATASET_FILES` chứa đúng tên các file `datasetX_train_std.csv` đã được tạo.
    * Chạy:
    ```bash
    python3 training.py
    ```

4.  **Xem Báo cáo Đánh giá Hiệu suất:**
    ```bash
    python3 evaluation_reporter.py
    ```
    Dựa vào kết quả này để chọn model tốt nhất cho website.

5.  **Chạy Ứng dụng Web Demo:**
    * Mở file `app.py` và cập nhật biến `DATASET_FOR_WEB` thành tên ngắn gọn của file huấn luyện đã cho model tốt nhất (ví dụ: `DATASET_FOR_WEB = "dataset4_train_std"`). Đảm bảo `model_types_for_web` bao gồm các model bạn muốn (RandomForest, NaiveBayes, XGBoost).
    * Chạy:
    ```bash
    python3 app.py
    ```
    Mở trình duyệt và truy cập: `http://127.0.0.1:5000`

---
```markdown
## Cấu trúc Thư mục chính
```
SQL_INJECTION_DETECTOR/
├── data/
│   ├── dataset1.csv                     # File dataset gốc 1
│   ├── dataset2.csv                     # File dataset gốc 2
│   ├── dataset3.csv                     # File dataset gốc 3
│   ├── dataset4.csv                     # File dataset gốc 4
│   ├── dataset1_train_std.csv           # Dữ liệu huấn luyện đã chuẩn hóa từ dataset1
│   ├── dataset2_train_std.csv           # Dữ liệu huấn luyện đã chuẩn hóa từ dataset2
│   ├── dataset3_train_std.csv           # Dữ liệu huấn luyện đã chuẩn hóa từ dataset3
│   ├── dataset4_train_std.csv           # Dữ liệu huấn luyện đã chuẩn hóa từ dataset4
│   └── datatest.csv                     # Bộ dataset kiểm thử chung, đã chuẩn hóa
├── model/
│   ├── vectorizer_datasetX_train_std.pkl
│   ├── RandomForest_datasetX_train_std.pkl
│   ├── NaiveBayes_datasetX_train_std.pkl
│   ├── XGBoost_datasetX_train_std.pkl
│   └── evaluation_results_on_common_test.json
├── static/images/
│   ├── bg.jpg
│   └── uit-logo.png
├── templates/index.html
├── evaluation_reports/
│   ├── formatted_eval_table_Accuracy.csv
│   ├── formatted_eval_table_Precision.csv
│   ├── formatted_eval_table_Recall.csv
│   └── formatted_eval_table_F1-Score.csv
├── app.py                     # Backend ứng dụng web Flask
├── training.py                # Script huấn luyện model và đánh giá trên common test
├── evaluation_reporter.py     # Script tạo bảng báo cáo hiệu suất
├── prepare_common_test_set.py # Script chuẩn bị dữ liệu từ file gốc
├── requirements.txt           # Danh sách thư viện 
└── README.md                  # File hướng dẫn
    ```
---

## Dataset

* **Dataset Gốc (trong `data/`):** 4 file (`dataset1.csv` đến `dataset4.csv`) do người dùng cung cấp. Chúng có thể có header và delimiter khác nhau (cần cấu hình trong `prepare_common_test_set.py`).
* **Dataset đã Chuẩn hóa (được tạo bởi `prepare_common_test_set.py` trong `data/`):**
    * **File huấn luyện (`datasetX_train_std.csv`):** Được tạo ra từ 80% dữ liệu của file dataset gốc tương ứng.
    * **File kiểm thử chung (`datatest.csv`):** Được tạo ra từ việc gộp 20% dữ liệu của mỗi file dataset gốc.
    * **Đặc điểm các file đã chuẩn hóa:** Không có header, delimiter là dấu chấm phẩy (`;`), 2 cột (cột 1: `query`, cột 2: `label` với `1` cho SQLi, `0` cho bình thường).

---

## Mô tả Model

* **Thuật toán sử dụng:** Random Forest, Naive Bayes (`MultinomialNB`), và XGBoost (`XGBClassifier`).
* **Tiền xử lý dữ liệu:** Các câu truy vấn SQL được chuyển đổi thành vector số bằng cách sử dụng `CountVectorizer` từ thư viện `scikit-learn`, với các tham số `min_df=2` và `max_df=0.8`.
* **Huấn luyện:** Model được huấn luyện trên các file `datasetX_train_std.csv` (là 80% dữ liệu từ mỗi file gốc, đã chuẩn hóa).
* **Đánh giá:** Tất cả các model đã huấn luyện được đánh giá trên cùng một file `datatest.csv` chung.
* **Lựa chọn model cho Web App:** Dựa trên kết quả đánh giá trên `datatest.csv`, người dùng cập nhật biến `DATASET_FOR_WEB` trong `app.py` (ví dụ: `"dataset4_train_std"` nếu model huấn luyện trên dataset4_train_std cho kết quả tốt nhất).

---

## Các Thư viện Chính Đã Sử Dụng

* Flask
* scikit-learn
* pandas
* joblib
* json
* xgboost

Phiên bản cụ thể của các thư viện được liệt kê trong file `requirements.txt`.
