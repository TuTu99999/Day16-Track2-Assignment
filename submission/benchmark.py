import gc
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path("creditcard.csv")
RESULT_PATH = Path("benchmark_result.json")

SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE_OF_REMAINDER = 0.25
LATENCY_REPEATS = 100
THROUGHPUT_REPEATS = 10

# Theo cấu hình mặc định của repo.
INSTANCE_TYPE = "t3.medium"
REGION = "us-east-1"


# 1. Load dataset
load_start = time.perf_counter()
df = pd.read_csv(DATA_PATH)
data_load_seconds = time.perf_counter() - load_start

if "Class" not in df.columns:
    raise ValueError("Dataset không có cột target 'Class'")

class_counts = {
    str(label): int(count)
    for label, count in df["Class"].value_counts().sort_index().items()
}

X = df.drop(columns=["Class"])
y = df["Class"].astype(np.int8)

# 2. Chia 80% train+validation và 20% test.
X_train_valid, X_test, y_train_valid, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y,
)

# 25% của phần 80% còn lại = 20% toàn bộ dataset.
# Kết quả cuối: 60% train, 20% validation, 20% test.
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_valid,
    y_train_valid,
    test_size=VALIDATION_SIZE_OF_REMAINDER,
    random_state=SEED,
    stratify=y_train_valid,
)

# Giải phóng các DataFrame trung gian để tiết kiệm RAM.
del df, X, y, X_train_valid, y_train_valid
gc.collect()

# 3. Train với validation để early stopping.
model = LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=SEED,
    n_jobs=-1,
    force_col_wise=True,
    verbosity=-1,
)

training_start = time.perf_counter()
model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)
training_seconds = time.perf_counter() - training_start

best_iteration = getattr(model, "best_iteration_", None)
if not best_iteration:
    best_iteration = model.get_params()["n_estimators"]

# 4. Đánh giá duy nhất trên test set.
test_probabilities = model.predict_proba(
    X_test,
    num_iteration=best_iteration,
)[:, 1]

test_predictions = (test_probabilities >= 0.5).astype(np.int8)

metrics = {
    "auc_roc": float(roc_auc_score(y_test, test_probabilities)),
    "accuracy": float(accuracy_score(y_test, test_predictions)),
    "f1": float(f1_score(y_test, test_predictions, zero_division=0)),
    "precision": float(
        precision_score(y_test, test_predictions, zero_division=0)
    ),
    "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
}

# 5. Đo latency cho một dòng.
# Dữ liệu đã nằm trong RAM; phép đo chỉ bao gồm predict_proba().
one_row = X_test.iloc[[0]]

for _ in range(5):
    model.predict_proba(one_row, num_iteration=best_iteration)

latency_samples_ms = []
for _ in range(LATENCY_REPEATS):
    start = time.perf_counter()
    model.predict_proba(one_row, num_iteration=best_iteration)
    latency_samples_ms.append((time.perf_counter() - start) * 1000)

latency_1_row_ms = statistics.median(latency_samples_ms)

# 6. Đo throughput cho batch 1.000 dòng.
batch = X_test.iloc[:1000]
model.predict_proba(batch, num_iteration=best_iteration)

throughput_start = time.perf_counter()
for _ in range(THROUGHPUT_REPEATS):
    model.predict_proba(batch, num_iteration=best_iteration)
throughput_seconds = time.perf_counter() - throughput_start

throughput_rows_per_second = (
    len(batch) * THROUGHPUT_REPEATS / throughput_seconds
)

# 7. Lưu kết quả.
result = {
    "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    "cloud": "AWS",
    "region": REGION,
    "instance_type": INSTANCE_TYPE,
    "python_version": platform.python_version(),
    "lightgbm_version": lgb.__version__,
    "scikit_learn_version": sklearn.__version__,
    "seed": SEED,
    "dataset": {
        "file": str(DATA_PATH),
        "rows": 284807,
        "columns": 31,
        "class_counts": class_counts,
    },
    "split": {
        "train_rows": len(X_train),
        "validation_rows": len(X_valid),
        "test_rows": len(X_test),
        "ratio": "60/20/20",
        "stratified": True,
    },
    "data_load_seconds": data_load_seconds,
    "training_seconds": training_seconds,
    "best_iteration": int(best_iteration),
    **metrics,
    "latency_1_row_ms": latency_1_row_ms,
    "latency_statistic": "median",
    "latency_repeat_count": LATENCY_REPEATS,
    "throughput_1000_rows_per_second": throughput_rows_per_second,
    "throughput_repeat_count": THROUGHPUT_REPEATS,
    "inference_scope": (
        "predict_proba only; data already loaded in memory; "
        "CSV loading excluded"
    ),
}

RESULT_PATH.write_text(
    json.dumps(result, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps(result, indent=2, ensure_ascii=False))
print(f"\nSaved result to: {RESULT_PATH.resolve()}")
