import cv2

image = cv2.imread(r"C:\Users\DELL\OneDrive\Desktop\riya doc\GOOD PARTS\memories\WhatsApp Image 2025-09-29 at 21.57.29_3f72c2a6 - Copy.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

cv2.imshow("grayimage", image)
cv2.waitKey(0)
cv2.destroyAllWindows()