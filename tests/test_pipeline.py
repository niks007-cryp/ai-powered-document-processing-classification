from pipeline import make_dataset, train, extract_fields


def test_dataset_balanced():
    rows = make_dataset(20, 42)
    labels = [r[1] for r in rows]
    assert len(rows) == 80 and len(set(labels)) == 4


def test_model_quality():
    _, metrics = train(42)
    assert metrics["accuracy"] >= 0.95 and metrics["macro_f1"] >= 0.95


def test_field_extraction():
    x = extract_fields("Invoice D1001 from Acme Supplies. Invoice date 2026-08-10. Total amount INR 12,500. Bill to Aarav Mehta.")
    assert x["document_id"] == "D1001" and x["amount_inr"] == "12,500" and x["customer"] == "Aarav Mehta"
