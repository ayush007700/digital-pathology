from src.pipeline.factory import build_pipeline


def test_pipeline_creation(dummy_model):

    pipeline = build_pipeline(
        dummy_model,
        "cpu",
    )

    assert pipeline is not None