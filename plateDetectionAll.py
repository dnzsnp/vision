import os 
import matplotlib.pyplot as plt 
import cv2 as cv
import numpy as np 
from pathlib import Path 



plate_path = Path("photos/plates")
croppedplate_path = Path("photos/cplates3")
head,tail = os.path.split(plate_path)
def segmentation(img):
     img_grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
     denoised = cv.bilateralFilter(img_grey,10,75,75)
     edges = cv.Canny(denoised,25,150)
     contours,hierarchy = cv.findContours(edges,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
     draw_contours = cv.drawContours(img,contours,-1,(0,0,0),1)
     return contours

def order_points(pts):
     rect = np.zeros([4,2],dtype=np.float32)
     s = pts.sum(axis=1) # x+y 
     rect[0] = pts[np.argmin(s)]
     rect[2] = pts[np.argmax(s)]

     diff = np.diff(pts,axis=1)
     rect[1]= pts[np.argmin(diff)]  
     rect[3]= pts[np.argmax(diff)]  
     return rect 







def findPlate(contours,img):
     img_area = img.shape[0] * img.shape[1]
     best_score = None
     best_approx = None 
     plate_idx=0
     plate_approx =None

     for c in contours:
          peri = cv.arcLength(c, True)          # konturun çevresi
          approx = cv.approxPolyDP(c, 0.02 * peri, True)   # sadeleştirilmiş şekil
           
          if len(approx) != 4 :
               continue
               
          x,y,w,h = cv.boundingRect(approx)
          
          if h==0:
                continue
          ratio = (w/h)
          area_ratio = (w*h)/img_area
          print(f"ratio:{ratio},area ratio:{area_ratio}")
          

          if(4.3 <= ratio <= 5.5  and 0.0001 <= area_ratio <= 0.006):
                    roi_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)[y:y+h,x:x+w]
                    mean_brightness = np.mean(roi_gray)
                    print(f"ratio:{ratio}, area_ratio:{area_ratio}, brightness:{mean_brightness}")
                    if mean_brightness > 140:
                         score = abs(ratio -4.9)
                         if best_score is None or score < best_score:
                                   print(f"mean brightness:{mean_brightness}")
                                   best_score = score
                                   best_approx = approx
                                   best_box = (x,y,w,h)
                        
     if best_approx is None:
                   print("Plate not found.")
                   return None                         
         
     x,y,w,h = best_box
     #pts = plate_approx.reshape(4,2).astype(np.float32)
     #app_pts = order_points(pts)
     #n1 = cv.drawContours(img,top10,plate_idx,(255,0,0),3)
     cropped = img[y:y+h, x:x+w]
     return cropped
     cv.imshow("plate",cropped)
     print("cropped size:",cropped.shape)

     print("x:",x,"\ny:",y,"\nw:",w,"\nh:",h)
     
     cv.imwrite(croppedplate_path+tail,cropped)                      
               
    

found = 0
skipped = 0


for i in list(plate_path.iterdir()):
    
    img = cv.imread(i)
    contours = segmentation(img)
    cropped = findPlate(contours,img)
    if cropped is not None:
         cv.imwrite(str(croppedplate_path/i.name),cropped)
         found +=1 
         
    else:
         skipped +=1
         


print("found:",found)
print("skipped:",skipped)
   
    
