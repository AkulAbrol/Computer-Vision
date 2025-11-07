import cv2
import numpy as np

image = cv2.imread('document.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Find the biggest rectangle
contours = sorted(contours, key=cv2.contourArea, reverse=True)
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
    if len(approx) == 4:
        target = approx
        break

# Perspective transform
pts1 = np.float32(target)
pts2 = np.float32([[0,0],[400,0],[400,400],[0,400]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
output = cv2.warpPerspective(image, matrix, (400,400))

cv2.imshow("Scanned", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
