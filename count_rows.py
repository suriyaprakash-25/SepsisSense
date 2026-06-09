import glob
import os

data_dir = r'd:\Vscode\SepsisSense\physionet.org\files\challenge-2019\1.0.0\training'
files = glob.glob(os.path.join(data_dir, '**', '*.psv'), recursive=True)

total_rows = 0
for filepath in files:
    with open(filepath, 'r') as fh:
        count = sum(1 for _ in fh)
    total_rows += count - 1  # subtract header row

train_rows     = round(total_rows * 0.8)
test_rows      = round(total_rows * 0.2)
train_patients = round(len(files) * 0.8)
test_patients  = round(len(files) * 0.2)

print(f"Total patient files  : {len(files):,}")
print(f"Total hourly records : {total_rows:,}")
print(f"")
print(f"--- 80/20 Stratified Split ---")
print(f"Train patients       : {train_patients:,}")
print(f"Train rows (records) : {train_rows:,}")
print(f"Test  patients       : {test_patients:,}")
print(f"Test  rows (records) : {test_rows:,}")
