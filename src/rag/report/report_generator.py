"""
Clinical Report Generator
"""

from datetime import datetime
from pathlib import Path

# Define output directory at module level or inside method
OUTPUT_DIR = Path("outputs/reports")


class ReportGenerator:

    def generate(
        self,
        prediction: str,
        confidence: float,
        llm_answer: str,
        model_name: str,
    ):
        # 1. Ensure directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # 2. Build Markdown content
        report = f"""
# Digital Pathology Report

Generated:
{datetime.now()}

---------------------------------------

Model

{model_name}

---------------------------------------

Prediction

{prediction}

---------------------------------------

Confidence

{confidence:.2f} %

---------------------------------------

Clinical Findings

{llm_answer}

---------------------------------------

Disclaimer

This report is AI-assisted and must be reviewed by a qualified pathologist.

"""

        # 3. Write report to disk
        filename = OUTPUT_DIR / "report.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)

        # 4. Return saved file path
        return filename