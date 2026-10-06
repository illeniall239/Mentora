def expected_calibration_error(confidences: list[float], correct: list[bool], n_bins: int) -> float:
    """Sum over non-empty bins of (|B|/N) * |acc(B) - conf(B)| with bins (b/n, (b+1)/n]; 0.0 goes in bin 0."""
    raise NotImplementedError
