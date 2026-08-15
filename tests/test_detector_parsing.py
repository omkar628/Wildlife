from backend.detector.classes import CLASS_ID_TO_NAME
from backend.detector.parser import parse_ultralytics_result, parse_yolo_result, xyxy_to_xywh


def test_xyxy_to_xywh():
    x, y, w, h = xyxy_to_xywh(10, 20, 50, 80)
    assert (x, y, w, h) == (10, 20, 40, 60)


def test_parse_yolo_result_maps_classes():
    detections = parse_yolo_result(
        boxes_xyxy=[[5, 6, 25, 46]],
        class_ids=[0],
        confidences=[0.88],
        class_map=CLASS_ID_TO_NAME,
    )
    assert len(detections) == 1
    item = detections[0]
    assert item.class_name == "tiger"
    assert item.class_id == 0
    assert item.confidence == 0.88
    assert item.bbox_width == 20
    assert item.bbox_height == 40


def test_parse_dict_result():
    detections = parse_ultralytics_result(
        {
            "xyxy": [[0, 0, 10, 10], [1, 2, 4, 8]],
            "cls": [1, 3],
            "conf": [0.7, 0.2],
        },
        CLASS_ID_TO_NAME,
    )
    assert [item.class_name for item in detections] == ["prey", "human"]
    assert detections[1].bbox_x == 1
    assert detections[1].bbox_height == 6
