from src.config import NUMERIC_FEATURES, TRAIN_END, VAL_END, TEST_END


def test_feature_contract_and_time_order():
    assert "lag_168" in NUMERIC_FEATURES
    assert TRAIN_END < VAL_END < TEST_END
