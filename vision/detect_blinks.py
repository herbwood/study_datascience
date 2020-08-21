from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	C = dist.euclidean(eye[0], eye[3])

	ear = (A + B) / (2.0 * C)

	return ear

def preprocessed_frame(vs, width=400, color=cv2.COLOR_BGR2GRAY):
    frame = vs.read()
    frame = imutils.resize(frame, width=width)
    gray = cv2.cvtColor(frame, color)
    return gray

def detector(gray, upsample=1):
    detector = dlib.get_frontal_face_detector()
    rects = detector(gray, upsample)
    return rects

def predictor(gray, rect, shape_predictor='shape_predictor_68_face_landmarks.dat'):
    predictor = dlib.shape_predictor(shape_predictor)
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    return shape

def blink_detector(frame, rects, EYE_AR_THRESH = 0.3, EYE_AR_CONSEC_FRAMES = 3, COUNTER = 0, TOTAL=0):
	for rect in rects:
		shape = predictor(frame, rect)
		(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


		# 왼쪽 및 오른쪽 눈 좌표를 추출한 다음 좌표를 사용하여 두 눈의 눈 종횡비 계산
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# 두 눈의 평균 눈 종횡비
		ear = (leftEAR + rightEAR) / 2.0

		# 왼쪽 눈과 오른쪽 눈의 눈꺼플을 계산 한 다음 각 눈을 시각화
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# 눈의 종횡비가 깜박임 임계값 보다 낮은지 확인하고, 그렇다면 눈 깜박임 프레임 카운터를 늘림
		if ear < EYE_AR_THRESH:
			COUNTER += 1

		# 그렇지 않으면, 눈의 종횡비가 깜박임 임계값 보다 낮지 않음
		else:
			# 눈의 깜박임 수가 연속 깜박임 프레임 임계값 보다 큰 경우 총 깜박임 횟수 증가
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1

			# 눈 깜박임 프레임 카운터 재설정
			COUNTER = 0

		# 프레임의 계산 된 눈 종횡비와 함께 프레임의 총 깜박임 수 표시
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		
	return frame


def main():
    print("[INFO] starting video stream thread...")
    fileStream = False
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    while True:
        if fileStream and not vs.more():
            break

        frame = preprocessed_frame(vs)
        rects = detector(frame, 0)
        frame = blink_detector(frame, rects)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    
        # 'q' key 를 누르면 루프 탈출
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
