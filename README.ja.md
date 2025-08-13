# 機械学習によるSQLインジェクション検知システム

本プロジェクトは、**Machine Learning** アルゴリズム（Random Forest、Naive Bayes、XGBoost）を用いて、SQLクエリから **SQLインジェクション（SQLi）** 攻撃の可能性を検知するシステムを実装しています。複数ソースからのデータ準備、モデル学習、共通テストセットでの評価、そしてリアルタイム検知のための **Flask** ベースWebアプリデモまで、包括的なワークフローを含みます。

**コース情報:**
* **大学:** University of Information Technology - VNUHCM
* **講義:** Introduction to Information Assurance and Security - IE105.P21
* **指導教員:** Dr. Nguyen Tan Cam
* **学生:**
    * Nguyen Dang Quang Phuc - 23521204
    * Tran Thi Nhu Phuong - 23521249

---

## 🚀 主な特徴

* **マルチソースデータ対応:** 区切り文字やヘッダー構造が異なる複数のソースファイルからデータを読み込み、標準化します。
* **堅牢な学習と評価:** 複数のMLモデル（Random Forest、Naive Bayes、XGBoost）を異なる訓練セットで学習し、統一された共通テストセットで評価します。
* **自動レポート生成:** 学習済みモデル全ての性能比較レポート（Accuracy、Precision、Recall、F1-Score）を生成します。
* **インタラクティブWebデモ:** **Flask** ベースのWebアプリで、ユーザーが入力したSQLクエリを使って最良モデルをテスト可能。

---

## 📋 プロジェクトワークフロー

1. **ソースデータ準備:** 4つのソースデータセットファイル（例: `dataset1.csv`, `dataset2.csv` など）を `data/` ディレクトリに配置。
2. **データ標準化と分割（`prepare_common_test_set.py`）:**
    * 4つのソースファイルを読み込み、異なるフォーマットを処理。
    * 各ソースから20%を抽出し、共通テストセット `datatest.csv` を作成。
    * 残り80%は標準化済み訓練データ（例: `dataset1_train_std.csv`）として保存。
    * **標準化フォーマット:** ヘッダーなし、区切り文字はセミコロン（`;`）。
3. **モデル学習と評価（`training.py`）:**
    * 各 `*_train_std.csv` ファイルでMLモデルを学習。
    * 学習済みモデルを共通 `datatest.csv` で評価。
    * 評価結果を `model/evaluation_results_on_common_test.json` に保存。
    * モデルとベクトライザは `model/` ディレクトリに保存。
4. **性能レポート生成（`evaluation_reporter.py`）:**
    * JSONファイルから結果を読み込み、性能比較表を生成。
5. **Webアプリデモ（`app.py`）:**
    * 最良モデルとベクトライザをロードし、WebインターフェースでリアルタイムSQLi検知。

---

## 🛠️ インストール手順

1. **必要条件:**
    * Python 3.8+
    * pip（Pythonパッケージマネージャ）
    * Homebrew（macOSユーザーのみ、必要に応じてXGBoost用`libomp`をインストール）

2. **仮想環境の作成と有効化:**
    プロジェクトルートディレクトリで:
    ```bash
    python3 -m venv venv
    ```
    環境を有効化:
    * macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    * Windows:
        ```bash
        venv\Scripts\activate
        ```

3. **`libomp`のインストール（macOSのみ、XGBoostエラー時）:**
    ```bash
    brew install libomp
    ```

4. **必要ライブラリのインストール:**
    （仮想環境が有効化されていることを確認）
    ```bash
    pip3 install -r requirements.txt
    ```

---

## ⚙️ 実行方法

1. **ソースデータ準備:**
    * `data/` ディレクトリに4つのソースデータファイルを置く。
    * **重要:** `prepare_common_test_set.py` 内の `SOURCE_DATASET_FILES_CONFIG` 辞書を、ソースファイルのファイル名やプロパティ（区切り文字、ヘッダー、エンコーディング）に合わせて更新。

2. **データ準備スクリプトの実行（初回またはソースデータ変更時）:**
    ```bash
    python3 prepare_common_test_set.py
    ```
    標準化された `*_train_std.csv` と `datatest.csv` が `data/` に生成されます。

3. **モデル学習:**
    ```bash
    python3 training.py
    ```

4. **性能レポートの確認:**
    ```bash
    python3 evaluation_reporter.py
    ```
    このレポートでWebアプリ用の最良モデルを判断。

5. **Webアプリデモの実行:**
    * `app.py` 内の `DATASET_FOR_WEB` を最良モデルを生成した訓練セット名に設定（例: `DATASET_FOR_WEB = "dataset4_train_std"`）。
    * アプリを起動:
        ```bash
        python3 app.py
        ```
    * ブラウザで `http://127.0.0.1:5000` を開く。

---

## 📁 プロジェクト構成

```
SQL_INJECTION_DETECTOR/
├── data/
│   ├── dataset1.csv              # ソースデータセット1
│   ├── ... （他のソースファイル）
│   ├── dataset1_train_std.csv    # 標準化済み訓練データ
│   ├── ... （他の訓練ファイル）
│   └── datatest.csv              # 共通標準化テストセット
├── model/
│   ├── *.pkl                     # 保存されたベクトライザとモデル
│   └── evaluation_results_on_common_test.json
├── static/
├── templates/
├── evaluation_reports/         # 生成された性能比較CSVレポート
├── app.py                      # Flask Webアプリバックエンド
├── training.py                 # モデル学習と評価スクリプト
├── evaluation_reporter.py      # 性能レポート生成スクリプト
├── prepare_common_test_set.py  # データ標準化スクリプト
├── requirements.txt
└── README.md
```

---

## 🤖 モデル詳細

* **アルゴリズム:** Random Forest、Multinomial Naive Bayes、XGBoost（`XGBClassifier`）
* **特徴抽出:** SQLクエリをscikit-learnの`CountVectorizer`で数値ベクトルに変換
* **評価:** 共通テストセット（`datatest.csv`）で全モデルをベンチマークし、公平な性能比較を実施

---

## 📚 使用ライブラリ

* Flask
* Scikit-learn
* Pandas
* Joblib
* XGBoost

詳細バージョンは `requirements.txt` に記載。
