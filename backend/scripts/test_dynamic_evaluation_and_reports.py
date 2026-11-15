import os
import sys
import json
import logging
from pathlib import Path

# Add project root and backend to PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

from backend.interfaces.api.services.csv_evaluation_service import CsvEvaluationService
from backend.infrastructure.evaluation.pdf_report_generator import PdfReportGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_dynamic_reports")

def test_runs():
    service = CsvEvaluationService()

    # -------------------------------------------------------------
    # 1. Authoritative 20,000-row dataset run: pred_1794327770_2951fbed
    # -------------------------------------------------------------
    logger.info("=== Testing Primary 20,000-row Run: pred_1794327770_2951fbed ===")
    pred_id_20k = "pred_1794327770_2951fbed"
    
    # Test 'flagged' filter PDF
    res = service.get_prediction_pdf_report(pred_id_20k, filter_mode="flagged")
    assert res is not None, "Failed to generate 20k flagged report"
    pdf_20k_flagged, fn_20k_flagged = res
    out_20k_flagged_path = project_root / f"evaluation_report_{pred_id_20k}.pdf"
    out_20k_flagged_path.write_bytes(pdf_20k_flagged)
    logger.info(f"Saved 20k flagged PDF to {out_20k_flagged_path} ({len(pdf_20k_flagged):,} bytes, filename: {fn_20k_flagged})")

    # Test 'all' filter PDF (20,000 rows in chunked tables)
    res_all = service.get_prediction_pdf_report(pred_id_20k, filter_mode="all")
    assert res_all is not None, "Failed to generate 20k all report"
    pdf_20k_all, fn_20k_all = res_all
    out_20k_all_path = project_root / f"evaluation_report_{pred_id_20k}_all.pdf"
    out_20k_all_path.write_bytes(pdf_20k_all)
    logger.info(f"Saved 20k all-rows PDF to {out_20k_all_path} ({len(pdf_20k_all):,} bytes, filename: {fn_20k_all})")

    # -------------------------------------------------------------
    # 2. Secondary 500-row dataset run: demo_ibm_aml_500.csv
    # -------------------------------------------------------------
    logger.info("=== Evaluating Secondary 500-row Dataset: demo_ibm_aml_500.csv ===")
    csv_500_path = project_root / "data" / "demo" / "demo_ibm_aml_500.csv"
    if not csv_500_path.exists():
        raise FileNotFoundError(f"Missing demo file {csv_500_path}")
    
    csv_500_bytes = csv_500_path.read_bytes()
    res_500 = service.evaluate_csv(
        file_bytes=csv_500_bytes,
        filename="demo_ibm_aml_500.csv",
        model_name="hi_small_lstm_fedavg_iid_seed42",
        split="test",
        operating_point="frozen",
    )
    pred_id_500 = res_500.prediction_id or res_500.download_id
    logger.info(f"Evaluation succeeded for 500-row CSV. Run ID: {pred_id_500}, total_evaluated: {res_500.total_evaluated}")

    # Generate 500-row flagged PDF
    res_500_flagged = service.get_prediction_pdf_report(pred_id_500, filter_mode="flagged")
    assert res_500_flagged is not None, "Failed to generate 500-row flagged report"
    pdf_500_flagged, fn_500_flagged = res_500_flagged
    out_500_flagged_path = project_root / f"evaluation_report_{pred_id_500}.pdf"
    out_500_flagged_path.write_bytes(pdf_500_flagged)
    logger.info(f"Saved 500-row flagged PDF to {out_500_flagged_path} ({len(pdf_500_flagged):,} bytes, filename: {fn_500_flagged})")

    # Generate 500-row all-rows PDF
    res_500_all = service.get_prediction_pdf_report(pred_id_500, filter_mode="all")
    assert res_500_all is not None, "Failed to generate 500-row all report"
    pdf_500_all, fn_500_all = res_500_all
    out_500_all_path = project_root / f"evaluation_report_{pred_id_500}_all.pdf"
    out_500_all_path.write_bytes(pdf_500_all)
    logger.info(f"Saved 500-row all-rows PDF to {out_500_all_path} ({len(pdf_500_all):,} bytes, filename: {fn_500_all})")

    print("\n================ REPORT VERIFICATION SUMMARY ================")
    print(f"Run 1 (20,000-row): ID = {pred_id_20k}")
    print(f"  Flagged Report: {out_20k_flagged_path.name} ({len(pdf_20k_flagged):,} bytes)")
    print(f"  All-Rows Report: {out_20k_all_path.name} ({len(pdf_20k_all):,} bytes)")
    print(f"Run 2 (500-row):    ID = {pred_id_500}")
    print(f"  Flagged Report: {out_500_flagged_path.name} ({len(pdf_500_flagged):,} bytes)")
    print(f"  All-Rows Report: {out_500_all_path.name} ({len(pdf_500_all):,} bytes)")
    print("=============================================================")

if __name__ == "__main__":
    test_runs()
