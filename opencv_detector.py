import cv2
import numpy as np

# YOLO 모델 파일 경로 설정
config_path = "test_before_cnn/yolov3.cfg"
weights_path = "test_before_cnn/yolov3.weights"
names_path = "test_before_cnn/coco.names"

# COCO 클래스 이름 로드
with open(names_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# OpenCV DNN 모듈을 사용해 YOLO 모델 로드
net = cv2.dnn.readNet(weights_path, config_path)

# 카메라 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 입력을 위한 전처리 (크기 조정 및 Blob 생성)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # YOLO 모델 예측
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)

    h, w = frame.shape[:2]

    # 탐지 결과를 화면에 표시
    boxes, confidences, class_ids = [], [], []
    for detection in detections:
        for object_detection in detection:
            scores = object_detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # 신뢰도 0.5 이상일 때만 Bounding Box 그리기
            if confidence > 0.5:
                box = object_detection[0:4] * np.array([w, h, w, h])
                center_x, center_y, width, height = box.astype("int")

                # Bounding Box의 좌상단 좌표 계산
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum Suppression을 사용해 중복된 박스 제거
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, score_threshold=0.5, nms_threshold=0.4
    )

    for i in indices:
        x, y, w, h = boxes[i]
        label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)

        # Bounding Box와 라벨 그리기
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 결과 프레임 보여주기
    cv2.imshow("YOLO Real-time Object Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
