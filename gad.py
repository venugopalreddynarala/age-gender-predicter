import cv2
import time
import argparse

def highlightFace(net, frame, conf_threshold=0.85):  # Higher threshold for better accuracy
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], swapRB=True, crop=False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frameOpencvDnn, faceBoxes


def preprocess_face(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    face_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return face_bgr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help="Image path or leave blank for webcam")
    args = parser.parse_args()

    # Model files
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    # Model config
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    # Load models
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    # Set backend/target (CPU for compatibility)
    for net in [faceNet, ageNet, genderNet]:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Open webcam or image
    video = cv2.VideoCapture(args.image if args.image else 0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    padding = 20
    last_check_time = 0
    interval = 2  # seconds

    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            break

        current_time = time.time()
        if current_time - last_check_time >= interval:
            last_check_time = current_time

            resultImg, faceBoxes = highlightFace(faceNet, frame)
            if not faceBoxes:
                print("No face detected")
            else:
                for faceBox in faceBoxes:
                    face = frame[max(0, faceBox[1] - padding): min(faceBox[3] + padding, frame.shape[0] - 1),
                                 max(0, faceBox[0] - padding): min(faceBox[2] + padding, frame.shape[1] - 1)]

                    face = preprocess_face(face)

                    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                    # Predict Gender
                    genderNet.setInput(blob)
                    genderPreds = genderNet.forward()
                    gender = genderList[genderPreds[0].argmax()]

                    # Predict Age
                    ageNet.setInput(blob)
                    agePreds = ageNet.forward()
                    age = ageList[agePreds[0].argmax()]

                    label = f'{gender}, {age}'
                    print(f'Detected: {label}')

                    # Draw label
                    cv2.putText(resultImg, label, (faceBox[0], faceBox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(resultImg, (faceBox[0], faceBox[1]), (faceBox[2], faceBox[3]), (0, 255, 0), 2)
        else:
            resultImg = frame

        cv2.imshow("Real-Time Age & Gender Detection", resultImg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
