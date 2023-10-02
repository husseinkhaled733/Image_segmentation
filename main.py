import cv2
import numpy as np
import imutils


def fixBorders(blended_frame):
    blended_frame = cv2.copyMakeBorder(blended_frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

    gray = cv2.cvtColor(blended_frame, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    mask = np.zeros(thresh_img.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    minRectangle = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        minRectangle = cv2.erode(minRectangle, None)
        sub = cv2.subtract(minRectangle, thresh_img)

    contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(areaOI)

    blended_frame = blended_frame[y:y + h, x:x + w]

    blended_frame = cv2.resize(blended_frame, (960, 540))

    return blended_frame


# Load the video files
cap1 = cv2.VideoCapture('Left (Better Quality).mp4')
cap2 = cv2.VideoCapture('Right(Better Quality).mp4')

width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
width = max(width1, width2)
height = max(height1, height2)
print(width, height)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('stitched2.mp4', fourcc, 30.0, (width, height))

# Create an image stitcher object
imageStitcher = cv2.Stitcher_create()

# Check if video files are opened successfully
if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open video files.")
    exit()

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        print("Error: Could not read frames from video files.")
        break

    # stitching frames
    status, blended_frame = imageStitcher.stitch((frame1, frame2))
    if not status:

        # fixing borders of stitched frame

        blended_frame = fixBorders(blended_frame)

        out.write(blended_frame)
        cv2.imshow('Stitched Video', blended_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()
