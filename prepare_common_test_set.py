# SQL_INJECTION_DETECTOR/prepare_common_test_set.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import io 

DATA_DIR = "data"
# Cấu hình các file nguồn 
# TẤT CẢ các file nguồn bây giờ đều có header và dùng delimiter ';'
SOURCE_DATASET_FILES_CONFIG = {
    "dataset1.csv": { "delimiter": ';', "header_row": 0, "encoding": "utf-8" }, 
    "dataset2.csv": { "delimiter": ';', "header_row": 0, "encoding": "utf-8" }, 
    "dataset3.csv": { "delimiter": ';', "header_row": 0, "encoding": "utf-8" }, 
    "dataset4.csv": { "delimiter": ';', "header_row": 0, "encoding": "utf-8" }  
}
# File test chung sẽ được tạo ra
COMMON_TEST_OUTPUT_FILE = "datatest.csv"
# Hậu tố cho file huấn luyện được tạo ra
TRAIN_FILE_SUFFIX = "_train_std.csv" 
# Delimiter mục tiêu cho các file output (KHÔNG HEADER)
TARGET_DELIMITER = ';' 
# Tỷ lệ dữ liệu trích cho test chung
TEST_SET_FRACTION = 0.20 
# Để đảm bảo kết quả chia nhất quán
RANDOM_STATE = 42 

def custom_data_loader(file_path, delimiter, header_row, specified_encoding):
    """Hàm đọc file CSV, tự động thử các encoding phổ biến nếu cần."""
    encodings_to_try = [specified_encoding, 'utf-8-sig', 'utf-16', 'latin1', 'iso-8859-1']
    df = None
    
    for enc in encodings_to_try:
        try:
            print(f"    Thử đọc file {os.path.basename(file_path)} với delimiter='{delimiter}' encoding: {enc} header_row: {header_row}")
            
            # quotechar='"': Xử lý trường hợp dữ liệu có dấu ngoặc kép bao quanh
            # skipinitialspace=True: Bỏ qua khoảng trắng thừa sau delimiter
            df_attempt = pd.read_csv(file_path, delimiter=delimiter, header=header_row, 
                                     on_bad_lines='skip', quotechar='"', escapechar='\\', 
                                     encoding=enc, skipinitialspace=True)
            df = df_attempt

            # Đảm bảo có đúng cột 'query' và 'label' sau khi đọc header
            # Pandas sẽ tự động dùng tên từ header nếu header_row=0
            if df is not None and ('query' not in df.columns or 'label' not in df.columns):
                # Nếu header_row=0 nhưng tên cột không phải 'query', 'label'
                if header_row == 0 and len(df.columns) >= 2:
                    print(f"      Cảnh báo: Header của {os.path.basename(file_path)} không phải 'query,label'. Sử dụng 2 cột đầu tiên và đặt tên là 'query', 'label'.")
                    df = df.iloc[:, :2].copy() # Lấy 2 cột đầu
                    df.columns = ['query', 'label'] # Gán tên chuẩn
                else:
                    print(f"      Lỗi: Không thể xác định cột 'query' và 'label' cho {os.path.basename(file_path)} với encoding {enc}.")
                    df = None 
                    continue # Thử encoding khác
            
            if df is not None:
                df.dropna(subset=['query', 'label'], inplace=True)
                if df.empty:
                    df = None 
                    continue 
                
                df['query'] = df['query'].astype(str)
                df['label'] = df['label'].astype(int)
                print(f"    Đọc thành công {os.path.basename(file_path)} với encoding: {enc}")
                return df 

        except UnicodeDecodeError:
            print(f"      Lỗi UnicodeDecodeError với encoding: {enc}. Đang thử encoding khác...")
            continue 
        except pd.errors.ParserError as pe:
            print(f"      Lỗi ParserError với file {os.path.basename(file_path)}: {pe}.")
            continue 
        except Exception as e:
            print(f"      Lỗi khác khi đọc {os.path.basename(file_path)} với encoding {enc}: {e}")
            df = None 
            continue 
                
    if df is None: 
        print(f"    Lỗi nghiêm trọng: Không thể đọc file {os.path.basename(file_path)} với bất kỳ encoding nào đã thử.")
    return df


def create_common_test_set_and_standardize_training_files():
    all_test_dfs = []
    
    print("--- Bắt đầu Tạo Bộ Test Chung và Chuẩn hóa File Huấn luyện ---")

    for source_filename, config in SOURCE_DATASET_FILES_CONFIG.items():
        file_path = os.path.join(DATA_DIR, source_filename)
        output_train_filename = f"{os.path.splitext(source_filename)[0]}{TRAIN_FILE_SUFFIX}" 
        output_train_path = os.path.join(DATA_DIR, output_train_filename)
        
        print(f"  Đang xử lý file nguồn: {source_filename} (cấu hình: {config})")
        
        df_original = custom_data_loader(file_path, 
                                         config['delimiter'], 
                                         config['header_row'],
                                         config.get('encoding', 'utf-8'))

        if df_original is None or df_original.empty:
            print(f"    LƯU Ý: File {source_filename} rỗng hoặc không đọc được đúng. Bỏ qua.")
            continue

        # Kiểm tra lại lần nữa sau khi custom_data_loader trả về
        if 'query' not in df_original.columns or 'label' not in df_original.columns:
            print(f"    LỖI: Sau khi tải, file {source_filename} không có cột 'query' hoặc 'label'. Bỏ qua.")
            continue

        if len(df_original) < 10: 
            print(f"    LƯU Ý: Dataset {source_filename} quá nhỏ ({len(df_original)} dòng). Toàn bộ sẽ được dùng để huấn luyện (đã chuẩn hóa), không trích cho test chung.")
            df_original.to_csv(output_train_path, sep=TARGET_DELIMITER, header=False, index=False) # Luôn lưu KHÔNG header
            print(f"    Đã lưu toàn bộ {source_filename} vào {output_train_filename} với delimiter '{TARGET_DELIMITER}' (không header).")
            continue

        df_remaining_train, df_for_common_test = train_test_split(
            df_original, 
            test_size=TEST_SET_FRACTION, 
            random_state=RANDOM_STATE,
            stratify=df_original['label'] if len(df_original['label'].unique()) > 1 and df_original['label'].value_counts().min() >= 2 else None
        )
        
        all_test_dfs.append(df_for_common_test)
        
        # Lưu phần huấn luyện với delimiter mục tiêu và KHÔNG có header
        df_remaining_train.to_csv(output_train_path, sep=TARGET_DELIMITER, header=False, index=False)
        print(f"    Đã trích {len(df_for_common_test)} mẫu cho bộ test chung.")
        print(f"    Đã lưu {len(df_remaining_train)} mẫu còn lại vào {output_train_filename} với delimiter '{TARGET_DELIMITER}' (không header).")

    if not all_test_dfs:
        print("Không có dữ liệu nào được trích để tạo bộ test chung. Dừng lại.")
        return

    common_test_df = pd.concat(all_test_dfs, ignore_index=True)
    common_test_df = common_test_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    common_test_output_path = os.path.join(DATA_DIR, COMMON_TEST_OUTPUT_FILE)
    try:
        # Lưu bộ test chung với delimiter mục tiêu và KHÔNG có header
        common_test_df.to_csv(common_test_output_path, sep=TARGET_DELIMITER, header=False, index=False)
        print(f"\nĐã tạo thành công Bộ Kiểm thử Chung: {common_test_output_path} ({len(common_test_df)} mẫu) với delimiter '{TARGET_DELIMITER}' (không header).")
    except Exception as e:
        print(f"LỖI khi lưu file {COMMON_TEST_OUTPUT_FILE}: {e}")

if __name__ == "__main__":
    create_common_test_set_and_standardize_training_files()
    print("\nHoàn tất! Kiểm tra thư mục 'data/' để thấy các file:")
    print(f"- '{COMMON_TEST_OUTPUT_FILE}' (bộ test chung, delimiter '{TARGET_DELIMITER}', không header).")
    print(f"- Các file '*{TRAIN_FILE_SUFFIX}' (dữ liệu huấn luyện đã chuẩn hóa delimiter thành '{TARGET_DELIMITER}', không header).")
    print("\nBước tiếp theo: Cập nhật file 'training.py' để sử dụng các file huấn luyện mới này và file test chung (tất cả đều không header, delimiter ';').")