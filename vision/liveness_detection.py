# USAGE
# blink_detection_demo.mp4 (자신의 눈 깜박임 영상을 촬영하여 blink_detection_demo.mp4 이름으로 저장 후 실행)
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import dlib
import cv2
import os

##### eye blink detection

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])

	ear = (A + B) / (2.0 * C)
    
	return ear

shape_predictor = "shape_predictor_68_face_landmarks.dat"
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0


print("[INFO] loading facial landmark predictor...")
eye_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]



##### liveness detection

model = "liveness.model"
le = "le.pickle"
face_detector = "face_detector"

print("[INFO] loading face detector...")
protoPath = os.path.sep.join([face_detector, "deploy.prototxt"])
modelPath = os.path.sep.join([face_detector, "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("[INFO] loading liveness detector...")
model = load_model(model)
le = pickle.loads(open(le, "rb").read())



##### 비디오 스트림을 초기화
print("[INFO] starting video stream...")
fileStream = False
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
	if fileStream and not vs.more():
		break

	# 스레드 비디오 파일 스트림에서 프레임을 가져 와서 크기를 조정한 다음 Grayscale 채널로 변환
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Grayscale 프레임에서 얼굴 감지
	rects = eye_detector(gray, 0)
    
	# 얼굴 감지 반복
	for rect in rects:
		# 얼굴 영역의 얼굴 랜드 마크를 결정한 다음 얼굴 랜드 마크 (x, y) 좌표를 NumPy 배열로 변환
		startX, startY, endX, endY = rect.left(), rect.top(), rect.right(), rect.bottom()
		shape = predictor(gray, rect)
		# print(shape)
		shape = face_utils.shape_to_np(shape)
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

		face = frame[startY:endY, startX:endX]
		face = cv2.resize(face, (32, 32))
		face = face.astype("float") / 255.0
		face = img_to_array(face)
		face = np.expand_dims(face, axis=0)

		preds = model.predict(face)[0]
		j = np.argmax(preds)
		label = le.classes_[j]
		print(label)

		# 눈의 종횡비가 깜박임 임계값 보다 낮은지 확인하고, 그렇다면 눈 깜박임 프레임 카운터를 늘림
		if ear < EYE_AR_THRESH:
			COUNTER += 1
			label = "{}: {:.4f}".format(label, preds[j])

		# 그렇지 않으면, 눈의 종횡비가 깜박임 임계값 보다 낮지 않음
		else:
			# 눈의 깜박임 수가 연속 깜박임 프레임 임계값 보다 큰 경우 총 깜박임 횟수 증가
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
				if label == 'real':
					label = "real : {:.4f}".format(preds[j])
				else:
					label = "fake : {:.4f}".format(preds[j])
					
			COUNTER = 0

		cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


	# 프레임 보여줌
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# 'q' key 를 누르면 루프 탈출
	if key == ord("q"):
		break

# Clean up
cv2.destroyAllWindows()
vs.stop()