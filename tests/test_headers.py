import pytest

from app.services.llm_service import align_profit_headers

# Header shape from the Arçelik pilot and amounts from eval/arcelik-dev.jsonl (q01/q04).
ROW = "|  | Net Dönem Kârı | -8.355.687.524,97 | -11.655.976.711,00 |"
ALIGNED = "| | | SPK’ya göre | Yasal Kayıtlara (YK) göre |"


def test_shifted_arcelik_header_matches_amount_columns():
    shifted = "| | SPK’ya göre | Yasal Kayıtlara (YK) göre | |"
    assert align_profit_headers(f"{shifted}\n{ROW}") == f"{ALIGNED}\n{ROW}"


@pytest.mark.parametrize("table", [f"{ALIGNED}\n{ROW}", ROW, "| Kalem | 2025 | 2024 |"])
def test_other_tables_are_preserved(table):
    assert align_profit_headers(table) == table
