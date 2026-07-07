from service.schemas import Detection, HealthResponse, PredictResponse


def test_detection_field_names():
    """Detection model has the documented field names."""
    d = Detection(
        id=0,
        class_id=1,
        class_name="text",
        bbox_xyxy=[10.0, 20.0, 110.0, 70.0],
        bbox_xywh=[60.0, 45.0, 100.0, 50.0],
        score=0.95,
    )
    assert d.id == 0
    assert d.class_name == "text"
    assert d.bbox_xyxy == [10.0, 20.0, 110.0, 70.0]
    assert d.score == 0.95


def test_predict_response_serialization():
    """PredictResponse dumps to camelCase-free JSON (snake_case) by default."""
    resp = PredictResponse(
        width=1240,
        height=1754,
        annotated_image="data:image/jpeg;base64,xxx",
        detections=[],
        num_detections=0,
        inference_time_ms=234,
        model_imgsz=1024,
        conf_threshold=0.2,
        device="cpu",
    )
    j = resp.model_dump()
    assert j["width"] == 1240
    assert j["annotated_image"] == "data:image/jpeg;base64,xxx"
    assert j["num_detections"] == 0
    assert j["inference_time_ms"] == 234
    assert j["device"] == "cpu"


def test_health_response_serialization():
    h = HealthResponse(
        status="ok",
        device="cpu",
        model_loaded=True,
        max_concurrent=1,
        model_path="/tmp/model.pt",
    )
    j = h.model_dump()
    assert j["status"] == "ok"
    assert j["model_loaded"] is True