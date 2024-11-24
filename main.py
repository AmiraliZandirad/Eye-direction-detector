import cv2 as cv
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector

detector = FaceDetector()
meshdetector = FaceMeshDetector(maxFaces=1)

left_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
right_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

current_direction = {"left_eye": None, "right_eye": None}

cam = cv.VideoCapture(0)

while cam.isOpened():
    ret, frame = cam.read()
    if ret:
        face_img, bbox = detector.findFaces(frame)
        face_img, faces = meshdetector.findFaceMesh(frame)

        if bbox and faces:
            face = faces[0]

            def process_eye(eye_indices, frame, face, eye_name):
                global current_direction
                eye_points = np.array([[face[p][0], face[p][1]] for p in eye_indices])
                (ex, ey, ew, eh) = cv.boundingRect(eye_points)
                eye_roi = frame[ey:ey + eh, ex:ex + ew]
                eye_roi_gr = cv.cvtColor(eye_roi, cv.COLOR_BGR2GRAY)
                _, iris = cv.threshold(eye_roi_gr, 40, 255, cv.THRESH_BINARY_INV)
                contours, _ = cv.findContours(iris, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

                if contours:
                    (ix, iy, iw, ih) = cv.boundingRect(contours[0])
                    ix_cntr_e = ix + int(iw / 2)
                    offset = 10
                    if ix_cntr_e > int(ew / 2) + offset:
                        new_direction = "right"
                    elif ix_cntr_e < int(ew / 2) - offset:
                        new_direction = "left"
                    else:
                        new_direction = "center"
                    
                    if current_direction[eye_name] != new_direction:
                        current_direction[eye_name] = new_direction
                        print(f"{eye_name}: {new_direction}")

                return current_direction[eye_name]

            left_status = process_eye(left_eye, frame, face, "left_eye")
            right_status = process_eye(right_eye, frame, face, "right_eye")

            if left_status:
                cv.putText(frame, f"Left Eye: {left_status}", (50, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if right_status:
                cv.putText(frame, f"Right Eye: {right_status}", (50, 100), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv.imshow('Eye Tracking', frame)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cam.release()
cv.destroyAllWindows()
