"""
SepsisSense Universal Data Harmonization & Schema Mapping Engine
===============================================================

Multi-center clinical data ingestion, schema harmonization, and standardisation
across diverse electronic health record (EHR) databases:
1. PhysioNet 2019 Computing in Cardiology Challenge (Sets A, B, and C)
2. MIMIC-IV Sepsis-3 Clinical Cohort
3. eICU Collaborative Research Database (eICU-CRD)
4. AmsterdamUMCdb & Open Clinical Sepsis Cohorts

Author: SepsisSense Machine Learning & Data Science Engineering Team
License: MIT
"""

import os
import sys
import glob
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ==============================================================================
# CANONICAL SCHEMA DEFINITIONS
# ==============================================================================

CANONICAL_VITALS: List[str] = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"
]

CANONICAL_LABS: List[str] = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2",
    "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
    "PTT", "WBC", "Fibrinogen", "Platelets", "EtCO2"
]

CANONICAL_DEMOGRAPHICS: List[str] = [
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"
]

CANONICAL_METADATA: List[str] = [
    "Patient", "DatasetSource", "SepsisLabel"
]

CANONICAL_SCHEMA: List[str] = (
    CANONICAL_METADATA + CANONICAL_DEMOGRAPHICS + CANONICAL_VITALS + CANONICAL_LABS
)

# Plausible physiological boundaries for data cleaning
PHYSIOLOGICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "HR": (20.0, 260.0),
    "O2Sat": (40.0, 100.0),
    "Temp": (28.0, 44.0),       # Celsius
    "SBP": (30.0, 280.0),
    "MAP": (20.0, 200.0),
    "DBP": (10.0, 180.0),
    "Resp": (4.0, 70.0),
    "Age": (18.0, 105.0),
    "Glucose": (10.0, 1200.0),
    "Lactate": (0.1, 30.0),
    "Creatinine": (0.1, 25.0),
    "WBC": (0.1, 150.0),
    "Platelets": (1.0, 1500.0),
    "pH": (6.7, 7.8),
}

# Fuzzy synonym mapping for external data sources
FEATURE_SYNONYMS: Dict[str, str] = {
    # Vitals
    "heartrate": "HR",
    "heart_rate": "HR",
    "pulse": "HR",
    "hr": "HR",
    "spo2": "O2Sat",
    "o2sat": "O2Sat",
    "o2_sat": "O2Sat",
    "oxygen_saturation": "O2Sat",
    "temp": "Temp",
    "temperature": "Temp",
    "temperature_c": "Temp",
    "temp_c": "Temp",
    "sysbp": "SBP",
    "systolic": "SBP",
    "sbp": "SBP",
    "systolic_bp": "SBP",
    "diabp": "DBP",
    "diastolic": "DBP",
    "dbp": "DBP",
    "diastolic_bp": "DBP",
    "meanbp": "MAP",
    "map": "MAP",
    "mean_arterial_pressure": "MAP",
    "resprate": "Resp",
    "respiratory_rate": "Resp",
    "resp": "Resp",
    "rr": "Resp",

    # Labs
    "wbc": "WBC",
    "wbc_count": "WBC",
    "leukocytes": "WBC",
    "platelets": "Platelets",
    "platelet_count": "Platelets",
    "plt": "Platelets",
    "creatinine": "Creatinine",
    "creat": "Creatinine",
    "lactate": "Lactate",
    "lactic_acid": "Lactate",
    "glucose": "Glucose",
    "glu": "Glucose",
    "hgb": "Hgb",
    "hemoglobin": "Hgb",
    "hct": "Hct",
    "hematocrit": "Hct",
    "bun": "BUN",
    "blood_urea_nitrogen": "BUN",
    "ph": "pH",
    "potassium": "Potassium",
    "k": "Potassium",
    "sodium": "Sodium",
    "na": "Sodium",
    "chloride": "Chloride",
    "cl": "Chloride",
    "calcium": "Calcium",
    "ca": "Calcium",
    "bicarbonate": "HCO3",
    "hco3": "HCO3",
    "baseexcess": "BaseExcess",
    "be": "BaseExcess",
    "fio2": "FiO2",
    "paco2": "PaCO2",
    "sao2": "SaO2",
    "ast": "AST",
    "sgot": "AST",
    "alkalinephos": "Alkalinephos",
    "alp": "Alkalinephos",
    "bilirubin_total": "Bilirubin_total",
    "bili_total": "Bilirubin_total",
    "bilirubin_direct": "Bilirubin_direct",
    "troponini": "TroponinI",
    "trop_i": "TroponinI",
    "ptt": "PTT",
    "fibrinogen": "Fibrinogen",
    "magnesium": "Magnesium",
    "mg": "Magnesium",
    "phosphate": "Phosphate",
    "etco2": "EtCO2",

    # Demographics & Metadata
    "age": "Age",
    "gender": "Gender",
    "sex": "Gender",
    "iculengthofstay": "ICULOS",
    "iculos": "ICULOS",
    "hospadmtime": "HospAdmTime",
    "hospital_admission_time": "HospAdmTime",
    "unit1": "Unit1",
    "unit2": "Unit2",
    "sepsislabel": "SepsisLabel",
    "sepsis_label": "SepsisLabel",
    "is_sepsis": "SepsisLabel",
    "target": "SepsisLabel",
    "patient": "Patient",
    "patient_id": "Patient",
    "stay_id": "Patient",
    "subject_id": "Patient",
}


# ==============================================================================
# DATA HARMONIZER CLASS
# ==============================================================================

class UniversalDataHarmonizer:
    """
    Standardises multi-center ICU sepsis records into the unified 45-feature
    canonical clinical schema with data hygiene, range verification, and caching.
    """

    def __init__(self, cache_dir: str = "data/processed"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Map heterogeneous column names to the canonical schema using the synonym dictionary."""
        df = df.copy()
        rename_map = {}
        for col in df.columns:
            normalized = col.strip().lower().replace(" ", "_").replace("-", "_")
            if normalized in FEATURE_SYNONYMS:
                rename_map[col] = FEATURE_SYNONYMS[normalized]
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
        return df

    @staticmethod
    def _clean_physiological_bounds(df: pd.DataFrame) -> pd.DataFrame:
        """Cap physiological measurements to realistic clinical bounds and convert units."""
        df = df.copy()

        # Temperature unit check: convert Fahrenheit to Celsius if > 45.0
        if "Temp" in df.columns:
            f_mask = df["Temp"] > 45.0
            if f_mask.any():
                df.loc[f_mask, "Temp"] = (df.loc[f_mask, "Temp"] - 32.0) * (5.0 / 9.0)

        # Approximate MAP if SBP and DBP exist but MAP is missing
        if "MAP" in df.columns and "SBP" in df.columns and "DBP" in df.columns:
            missing_map = df["MAP"].isna() & df["SBP"].notna() & df["DBP"].notna()
            if missing_map.any():
                df.loc[missing_map, "MAP"] = (
                    df.loc[missing_map, "SBP"] + 2.0 * df.loc[missing_map, "DBP"]
                ) / 3.0

        # Apply range sanity filters
        for col, (low, high) in PHYSIOLOGICAL_BOUNDS.items():
            if col in df.columns:
                invalid_mask = (df[col] < low) | (df[col] > high)
                if invalid_mask.any():
                    df.loc[invalid_mask, col] = np.nan

        # Normalize gender to 1 (Male) and 0 (Female)
        if "Gender" in df.columns and df["Gender"].dtype == object:
            df["Gender"] = df["Gender"].astype(str).str.upper().map(
                {"M": 1, "MALE": 1, "1": 1, "F": 0, "FEMALE": 0, "0": 0}
            ).fillna(0).astype(int)

        return df

    def harmonize_dataframe(
        self,
        df: pd.DataFrame,
        source_name: str = "Unknown",
        default_patient_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Harmonize an arbitrary clinical DataFrame into the canonical 45-feature schema.
        """
        # Step 1: Standardize column names
        df = self._standardize_columns(df)

        # Step 2: Ensure Patient identifier exists
        if "Patient" not in df.columns:
            df["Patient"] = default_patient_id or "UNKNOWN_PATIENT"

        # Step 3: Ensure DatasetSource exists
        if "DatasetSource" not in df.columns:
            df["DatasetSource"] = source_name

        # Step 4: Ensure SepsisLabel exists
        if "SepsisLabel" not in df.columns:
            df["SepsisLabel"] = 0

        # Step 5: Clean ranges and unit conversions
        df = self._clean_physiological_bounds(df)

        # Step 6: Ensure all canonical columns exist (fill missing with NaN)
        for col in CANONICAL_SCHEMA:
            if col not in df.columns:
                df[col] = np.nan

        # Select columns in canonical order
        return df[CANONICAL_SCHEMA].copy()

    def load_physionet_cohort(
        self,
        data_dir: str,
        subset_name: Optional[str] = None,
        max_patients: Optional[int] = None,
        n_workers: int = 16
    ) -> pd.DataFrame:
        """
        High-performance parallel ingestion of PhysioNet 2019 .psv patient files.
        """
        if subset_name:
            target_path = os.path.join(data_dir, subset_name, "*.psv")
            source_tag = f"PhysioNet_{subset_name}"
        else:
            target_path = os.path.join(data_dir, "**", "*.psv")
            source_tag = "PhysioNet_Challenge"

        files = glob.glob(target_path, recursive=True)
        if not files:
            raise FileNotFoundError(f"No .psv files found under pattern: {target_path}")

        if max_patients is not None and max_patients > 0:
            files = files[:max_patients]

        logger.info(f"Loading {len(files):,} patient files from {source_tag} with {n_workers} workers...")
        t0 = time.time()

        def _read_file(fp: str) -> pd.DataFrame:
            df_p = pd.read_csv(fp, sep="|")
            p_id = os.path.splitext(os.path.basename(fp))[0]
            # Infer Hospital system from folder structure if available
            parent_name = os.path.basename(os.path.dirname(fp))
            src = parent_name if "set" in parent_name.lower() else source_tag
            df_p["Patient"] = p_id
            df_p["DatasetSource"] = src
            return df_p

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            dfs = list(executor.map(_read_file, files))

        combined = pd.concat(dfs, ignore_index=True)
        elapsed = time.time() - t0
        logger.info(
            f"Successfully ingested {combined.shape[0]:,} rows ({len(files):,} patients) in {elapsed:.2f}s."
        )

        # Harmonize into canonical structure
        harmonized = self.harmonize_dataframe(combined, source_name=source_tag)
        return harmonized

    def load_or_build_combined_dataset(
        self,
        base_data_dir: str,
        cache_filename: str = "multidataset_sepsis_harmonized.parquet",
        max_patients_per_subset: Optional[int] = None,
        force_rebuild: bool = False
    ) -> pd.DataFrame:
        """
        Load cached multi-center dataset or build fresh from source directories.
        """
        cache_path = os.path.join(self.cache_dir, cache_filename)

        if os.path.exists(cache_path) and not force_rebuild:
            logger.info(f"Loading cached harmonized dataset from: {cache_path}")
            df = pd.read_parquet(cache_path)
            logger.info(f"Loaded {len(df):,} records ({df['Patient'].nunique():,} patients) from cache.")
            return df

        logger.info("Building multi-center harmonized dataset from source repositories...")

        subsets = ["training_setA", "training_setB"]
        dfs = []

        for sub in subsets:
            sub_dir = os.path.join(base_data_dir, sub)
            if os.path.exists(sub_dir):
                logger.info(f"Ingesting cohort: {sub}...")
                df_sub = self.load_physionet_cohort(
                    base_data_dir,
                    subset_name=sub,
                    max_patients=max_patients_per_subset
                )
                dfs.append(df_sub)

        if not dfs:
            # Fallback to general search
            df_all = self.load_physionet_cohort(
                base_data_dir,
                max_patients=max_patients_per_subset
            )
            dfs.append(df_all)

        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Multi-center harmonization complete. Shape: {combined_df.shape}")

        # Save to parquet cache
        try:
            combined_df.to_parquet(cache_path, index=False, compression="snappy")
            logger.info(f"Cached harmonized dataset saved to {cache_path} ({os.path.getsize(cache_path) / 1024**2:.1f} MB)")
        except Exception as e:
            logger.warning(f"Could not cache to Parquet ({e}). Returning DataFrame in-memory.")

        return combined_df

