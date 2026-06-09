# Downloading Training Data

The PhysioNet Challenge 2019 training data is not included in this repository due to its large size (over 1GB).

## How to Download the Data

### Option 1: Download from PhysioNet (Recommended)

1. Visit the PhysioNet Challenge 2019 page:
   https://physionet.org/content/challenge-2019/1.0.0/

2. Download the training data files:
   - training_setA.zip
   - training_setB.zip

3. Extract the files to:
   ```
   physionet.org/files/challenge-2019/1.0.0/training/
   ```

### Option 2: Use Demo Mode

The application includes a demo mode that generates sample data automatically when no training data is found. Simply run the app:

```bash
streamlit run app/streamlit_app.py
```

The preprocessing module will automatically create sample data in `data/processed/` if no PhysioNet data is detected.

## Data Structure

After downloading and extracting, your directory should look like:

```
physionet.org/
└── files/
    └── challenge-2019/
        └── 1.0.0/
            └── training/
                ├── training_setA/
                │   ├── p000001.psv
                │   ├── p000002.psv
                │   └── ...
                └── training_setB/
                    ├── p100001.psv
                    ├── p100002.psv
                    └── ...
```

## Training the Model with Full Data

Once you have downloaded the data:

```bash
# Preprocess the data
python src/preprocess.py

# Generate features
python src/feature_engineering.py

# Train the model
python src/train_model.py
```

## Note on Data License

The PhysioNet Challenge 2019 data is provided under the Open Data Commons Open Database License (ODbL). Please review the license terms at:
https://physionet.org/content/challenge-2019/view-license/1.0.0/
