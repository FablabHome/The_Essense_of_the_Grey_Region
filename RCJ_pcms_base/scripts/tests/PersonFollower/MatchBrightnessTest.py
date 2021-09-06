import cv2 as cv
import color_transfer
import numpy as np

cap = cv.VideoCapture(0)
initIMG = None

while cap.isOpened():
    _, frame = cap.read()
    if frame is None:
        continue

    if initIMG is not None:
        source = initIMG
        result = color_transfer.color_transfer(source, frame)

        images = np.hstack((source, frame, result))
        cv.imshow('source, target, result', images)
    else:
        cv.imshow('frame', frame)

    key = cv.waitKey(16)
    if key in [27, ord('q')]:
        break
    elif key == ord('c'):
        initIMG = frame.copy()
        initIMG = cv.convertScaleAbs(initIMG, 1, 3)
        cv.destroyWindow('frame')

cap.release()
cv.destroyAllWindows()
