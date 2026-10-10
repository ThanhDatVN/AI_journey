from fraud.monitoring import drift_report


def test_no_drift_on_held_out_data(metadata, splits):
    _, _, test = splits
    report = drift_report(metadata["reference_profile"], test)
    assert (report.psi < 0.05).all()


def test_detects_amount_shift(metadata, splits):
    _, _, test = splits
    shifted = test.assign(amount_vnd=test.amount_vnd * 3)
    report = drift_report(metadata["reference_profile"], shifted).set_index("feature")
    assert report.loc["log_amount", "status"] == "ALERT"
    assert report.drop(index="log_amount").psi.max() < 0.05
