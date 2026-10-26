import cv2
import numpy as np
file = input("Upload the file")

image = cv2.imread(file) # read image
if image is None:
    print("ERROR !")
    exit()

original = image.copy()

#convert to gray scale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 

# filtering to remove noise nad smooth 
blurred_image = cv2.GaussianBlur(gray, (5,5),0) 

#for detecting border and to seperate object
edge = cv2.Canny(blurred_image,50,150)

contours,hierarchy = cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea,reverse=True)

#for making boundary 
document_contour = None
image_area = image.shape[0] * image.shape[1]
for contour in contours:
    area = cv2.contourArea(contour)

    # Ignore very small contours
    if area < 0.2 * image_area:
        continue
    epsilon = 0.02 * cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour , epsilon , True)

    if(len(approx) == 4):
        document_contour = approx
        break
if(document_contour is None):
    print("WE cannot detect")
    cv2.imshow("Edges" ,edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()


# contour done then we need to draw the contour
detected =  image.copy()
cv2.drawContours(detected,[document_contour],-1,(0,255,0),3)
cv2.imshow("Detected Document " , detected)

#arranging all 4 corner 
points = document_contour.reshape(4,2)
def order_points(points):
    ordered = np.zeros((4,2), dtype=np.float32)
    #top left has smallest x+y
    ordered[0] = points[np.argmin(points.sum(axis=1))]
    #bottom right has largest x+y
    ordered[2] = points[np.argmax(points.sum(axis=1))]
    diff = points[:, 0] - points[:, 1]
    ordered[1] = points[np.argmax(diff)]#top right
    ordered[3] = points[np.argmin(diff)]#bottom left

    return ordered

ordered_points = order_points(points)
debug = image.copy()

for i, point in enumerate(ordered_points):

    x, y = point.astype(int)

    cv2.circle(debug, (x, y), 8, (0, 0, 255), -1)

    cv2.putText(
        debug,
        str(i),
        (x + 10, y + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

cv2.imshow("Ordered Points", debug)
top_left , top_right,  bottom_right ,bottom_left= ordered_points

#calculate output dimension
width_top = np.linalg.norm(top_right - top_left)
width_bottom = np.linalg.norm (bottom_right - bottom_left)
max_width = int(max(width_bottom,width_top))

height_left= np.linalg.norm(bottom_left - top_left)
height_right = np.linalg.norm(bottom_right - top_right)
max_height = int(max(height_left, height_right))

#create destination point 
destination = np.array([[0,0],[max_width-1,0],
                        [max_width-1, max_height-1],
                        [0,max_height-1]]
                       , dtype=np.float32)

#perspective transformation 

matrix = cv2.getPerspectiveTransform(ordered_points , destination)
warped = cv2.warpPerspective(image ,matrix,(max_width , max_height))

#connected document to grayscale 
wraped_gray = cv2.cvtColor(warped , cv2.COLOR_BGR2GRAY)

#adaptive threshold

scanned = cv2.adaptiveThreshold(wraped_gray, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,
                                11 , 2)

output_file = "scanned_document.jpg"

cv2.imwrite(output_file , scanned)
cv2.imshow("original image", original )
cv2.imshow("Detected Document", detected)
cv2.imshow("Ordered Points", debug)
cv2.imshow("Warped Document", warped)
cv2.imshow("scanned image" , scanned)

print("Document scanned successfully")
print("saved as " , output_file)
cv2.waitKey(0)
cv2.destroyAllWindows()
