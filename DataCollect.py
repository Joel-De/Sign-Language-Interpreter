import cv2
import pathlib
import time

data_dir = "Numbers/5/"
imgcount = 2000
finalres = (200, 200)

cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 480)
c = 0
n = 0

time.sleep(3)  # Get set up

print("ran")
while True:
    ret, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    num = cv2.Laplacian(image, cv2.CV_64F).var()
    cv2.imshow("window3", image)
    image = cv2.resize(image, finalres)
    if num > 5 and c % 1 == 0:
        cv2.imshow("windows", image)

        n += 1
        cv2.imwrite(data_dir + str(n) + ".jpg", image)
        print("Printed img" + str(n))
        image = cv2.resize(image, (64, 64))

    c += 1
    cv2.waitKey(1)
    if n > imgcount:
        break
