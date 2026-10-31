"""
SepsisSense Clinical Text NLP & Multimodal EHR Ingestion Module
==============================================================

Processes unstructured clinical notes (nursing assessments, ED triage notes,
physician progress notes) to disambiguate physiological sepsis mimics from
true acute systemic infection:

1. Clinical Negation-Aware Entity Extraction (NegEx principles)
2. Sepsis Prodromal & Organ Dysfunction Detection (altered mental status, mottled skin, rigors)
3. Infection Source Identification (pneumonia, urosepsis, abdominal, catheter-related)
4. Sepsis Mimic & Benign Etiology Detection (post-op pain, panic/anxiety, PT ambulation)
5. Palliative / Comfort Measures Only (CMO) Alert Suppression Filter
6. Multimodal Clinical Suspicion Score (0.0 to 1.0) & Decision Adjustment

Author: SepsisSense Machine Learning & Clinical Data Science Team
License: MIT
"""

import os
import sys
import re
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import numpy as np

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger("ClinicalNLP")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _f = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s", datefmt="%H:%M:%S")
    _h.setFormatter(_f)
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


class ClinicalTextAnalyzer:
    """
    Specialized medical NLP analyzer for bedside ICU/ED clinical notes.
    Extracts infection suspicion, qualitative organ dysfunction, and identifies
    benign physiological mimics to suppress false alarms.
    """

    # Clinical Negation Prefix Triggers (up to 5 words ahead)
    NEGATION_TRIGGERS = [
        r"\bno\b", r"\bnot\b", r"\bdenies\b", r"\bdenied\b", r"\bwithout\b",
        r"\bno evidence of\b", r"\bno signs of\b", r"\brule out\b", r"\bruled out\b",
        r"\bnegative for\b", r"\bunlikely\b", r"\bfree of\b"
    ]

    # Acute Infection & Sepsis Symptoms (Positive Evidence)
    SEPSIS_INDICATORS = {
        "altered_mental_status": [
            r"altered mental status", r"\bams\b", r"confusion", r"confused",
            r"lethargic", r"lethargy", r"somnolent", r"obtunded", r"disoriented", r"encephalopathy"
        ],
        "perfusion_failure": [
            r"mottled", r"mottling", r"cool extremities", r"cold clammy",
            r"delayed capillary refill", r"cap refill >", r"poor perfusion", r"cyanotic", r"cyanosis"
        ],
        "rigors_fever": [
            r"rigors", r"shivering", r"chills", r"spiking fever", r"febrile", r"hypothermic", r"sweating profusely"
        ],
        "suspected_infection_site": [
            r"pneumonia", r"productive cough", r"purulent sputum", r"consolidation",
            r"urosepsis", r"urinary tract infection", r"\buti\b", r"dysuria", r"cloudy urine",
            r"peritonitis", r"acute abdomen", r"cellulitis", r"erythema", r"wound purulence",
            r"catheter infection", r"line sepsis", r"bacteremia"
        ],
        "clinical_sepsis_suspicion": [
            r"suspect sepsis", r"septic", r"sepsis concern", r"starting vancomycin",
            r"broad-spectrum", r"blood cultures drawn", r"lactic acidosis", r"refractory hypotension"
        ]
    }

    # Non-Infectious Physiological Mimics (Benign Explanations of Tachycardia/Tachypnea)
    MIMIC_INDICATORS = {
        "postoperative_pain": [
            r"post-op pain", r"surgical pain", r"incision pain", r"pain at surgical site",
            r"pain level \d+/10", r"mild pain", r"moderate pain", r"given morphine", r"given fentanyl"
        ],
        "anxiety_panic": [
            r"anxious", r"anxiety", r"panic", r"hyperventilating", r"crying", r"agitated", r"reassured patient"
        ],
        "physical_exertion": [
            r"ambulating", r"walking in hallway", r"physical therapy", r"\bpt session\b",
            r"sitting in chair", r"transferred to chair", r"exercise"
        ],
        "normal_alert": [
            r"alert and oriented", r"\bao ?x ?[34]\b", r"resting comfortably", r"sleeping comfortably",
            r"lungs clear", r"clear to auscultation", r"vital signs stable", r"hemodynamically stable"
        ]
    }

    # Palliative / Goals of Care (Triggers Complete Suppression)
    PALLIATIVE_INDICATORS = [
        r"comfort measures only", r"\bcmo\b", r"do not resuscitate", r"\bdnr\b",
        r"palliative care", r"hospice", r"terminal extubation", r"comfort care"
    ]

    def __init__(self):
        # Compile regular expressions
        self.neg_regex = re.compile("|".join(self.NEGATION_TRIGGERS), re.IGNORECASE)
        
        self.compiled_sepsis = {}
        for cat, patterns in self.SEPSIS_INDICATORS.items():
            self.compiled_sepsis[cat] = re.compile("|".join(patterns), re.IGNORECASE)

        self.compiled_mimics = {}
        for cat, patterns in self.MIMIC_INDICATORS.items():
            self.compiled_mimics[cat] = re.compile("|".join(patterns), re.IGNORECASE)

        self.compiled_palliative = re.compile("|".join(self.PALLIATIVE_INDICATORS), re.IGNORECASE)

    def analyze_note(self, text: str) -> Dict[str, Any]:
        """
        Analyze a clinical note, nursing assessment, or ED triage narrative.
        
        Returns
        -------
        Dict with:
        - 'nlp_sepsis_score': float in [0.0, 1.0] (clinical infection suspicion)
        - 'mimic_score': float in [0.0, 1.0] (benign physiological confounder evidence)
        - 'is_palliative_suppressed': bool (true if CMO/DNR detected)
        - 'risk_delta': float in [-0.25, +0.25] (recommended adjustment to probability)
        - 'detected_sepsis_features': List[str]
        - 'detected_mimics': List[str]
        - 'summary_interpretation': str
        """
        if not text or not isinstance(text, str) or not text.strip():
            return {
                "nlp_sepsis_score": 0.50,
                "mimic_score": 0.0,
                "is_palliative_suppressed": False,
                "risk_delta": 0.0,
                "detected_sepsis_features": [],
                "detected_mimics": [],
                "summary_interpretation": "No clinical text provided. Relying solely on structured vital signs & labs."
            }

        # 1. Check Palliative / Goals of Care
        is_palliative = bool(self.compiled_palliative.search(text))

        # Split into sentences for negation scope bounding
        sentences = re.split(r"[.\n\r;]+", text)
        
        detected_sepsis: Set[str] = set()
        detected_mimics: Set[str] = set()

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue

            # Check sepsis indicators
            for cat, regex in self.compiled_sepsis.items():
                for match in regex.finditer(s_clean):
                    start = match.start()
                    # Check if preceded by negation within ~30 chars
                    prefix = s_clean[max(0, start - 35):start]
                    if not self.neg_regex.search(prefix):
                        detected_sepsis.add(match.group(0).lower())

            # Check mimic indicators
            for cat, regex in self.compiled_mimics.items():
                for match in regex.finditer(s_clean):
                    start = match.start()
                    prefix = s_clean[max(0, start - 35):start]
                    if not self.neg_regex.search(prefix):
                        detected_mimics.add(match.group(0).lower())

        # 2. Compute Scores
        n_sepsis = len(detected_sepsis)
        n_mimics = len(detected_mimics)

        # Base infection score scaled by findings count
        sepsis_score = min(1.0, n_sepsis * 0.25)
        mimic_score = min(1.0, n_mimics * 0.30)

        # Risk delta adjustment bounded in [-0.25, +0.25]
        risk_delta = (sepsis_score * 0.25) - (mimic_score * 0.25)
        risk_delta = float(np.clip(risk_delta, -0.25, 0.25))

        # 3. Clinical Interpretation Summary
        if is_palliative:
            summary = "Patient has active Goals of Care (CMO/DNR) documented. Sepsis alerting should be suppressed."
        elif n_sepsis > 0 and n_mimics == 0:
            summary = f"Strong clinical note infection corroboration ({', '.join(sorted(detected_sepsis))}). Increases confidence of true septic deterioration."
        elif n_mimics > 0 and n_sepsis == 0:
            summary = f"Clinical note documents physiological mimic ({', '.join(sorted(detected_mimics))}). Vital sign elevations explained by non-septic etiology; suppresses false alarm."
        elif n_sepsis > 0 and n_mimics > 0:
            summary = f"Mixed clinical presentation: Infection markers ({', '.join(sorted(detected_sepsis))}) present alongside confounders ({', '.join(sorted(detected_mimics))}). Close monitoring indicated."
        else:
            summary = "Clinical narrative is non-specific; structured time-series indicators dominate risk assessment."

        return {
            "nlp_sepsis_score": float(sepsis_score),
            "mimic_score": float(mimic_score),
            "is_palliative_suppressed": is_palliative,
            "risk_delta": float(risk_delta),
            "detected_sepsis_features": sorted(list(detected_sepsis)),
            "detected_mimics": sorted(list(detected_mimics)),
            "summary_interpretation": summary
        }
