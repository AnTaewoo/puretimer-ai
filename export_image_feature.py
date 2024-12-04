import cv2
import numpy as np

# 카메라 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 전처리 (모델 입력 크기로 조정 및 정규화)
    input_frame = cv2.resize(frame, (224, 224))
    input_frame = input_frame / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)

    # 모델 예측
    class_pred, bbox_pred = model.predict(input_frame)

    # Bounding Box 좌표를 원본 프레임 크기로 변환
    h, w, _ = frame.shape
    xmin = int(bbox_pred[0][0] * w)
    ymin = int(bbox_pred[0][1] * h)
    xmax = int(bbox_pred[0][2] * w)
    ymax = int(bbox_pred[0][3] * h)

    # Bounding Box와 클래스 정보 프레임에 표시
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    label = f"Class: {np.argmax(class_pred)}"
    cv2.putText(
        frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
    )

    # 결과 보여주기
    cv2.imshow("Real-time Localization", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
