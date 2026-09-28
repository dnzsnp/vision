import os 
import matplotlib.pyplot as plt 
import numpy as np 
import cv2 as cv 
from pathlib import Path
#For loop folder -> segmentation -> character sep. -> 

image_folder =Path("C:/Users/deniz/Desktop/plates")
plate_path = Path("C:/Users/deniz/Desktop/vision/photos/plates")
crop_path = "ocr/chars"

letters = []
char_coor = []

def folder_jpg_extract(folder):
    for img in folder.rglob("*.jpg"):
        image = cv.imread(str(img))
        print("loaded:", image is not None)

        output_path = Path(plate_path) / img.name
        success = cv.imwrite(str(output_path), image)
        print("saved:", success, "->", output_path)

#folder_jpg_extract(image_folder)

head,tail = os.path.split(plate_path)






def segmentation(img):
     img_grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
     blur = cv.GaussianBlur(img_grey,(5,5),3)
     thres_img = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
     kernel = np.ones((3,3), np.uint8)
     closed = cv.morphologyEx(thres_img, cv.MORPH_CLOSE, kernel, iterations=1)
     contours,hierarchy = cv.findContours(closed,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
     draw_contours = cv.drawContours(img,contours,-1,(0,0,0),1)
     top10 = sorted(contours,key=lambda c:cv.contourArea(c,False), reverse=True)[:10]
     return top10 


def crop_contours(contours):
     char_coor=[]
     for i , c in enumerate(contours):
        peri = cv.arcLength(c, True)          # konturun çevresi
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        x,y,w,h = cv.boundingRect(c)
        print((h/w))
        if ((h/w) >= 1.7 and (h/w) <= 3.0):
            char_coor.insert(i,[x,y,w,h])
            print(f"{crop_path}/{i}_{tail},coordinates:{x},{y},{w},{h}")

     char_coor_sorted =sorted(char_coor,key=lambda x: x[0])
     return char_coor_sorted 


def char_crop_save(char_list,file_path): # sort list save  
      for i,v in enumerate(char_list):
            plt.ion()
            x,y,w,h = char_list[i]
            char_crop = img[y:y+h, x:x+w]
            padded_charcrop = cv.copyMakeBorder(char_crop,2,2,2,2,cv.BORDER_CONSTANT)
            char_resized = cv.resize(padded_charcrop,(32,32))
            plt.subplot(121)
            plt.title(file_path)
            plt.imshow(char_resized)
            plt.show(block=False)
            char = input("Character?")
            if char == ",":
                 print(f"{crop_path}/{char}_{tail},unreadable!")
                 continue
            plt.close()
            folder = os.path.join(crop_path, char)
            if os.path.isdir(folder):
                  print("folder exists.")
            else:
                  os.mkdir(folder)      
            cv.imwrite(os.path.join(folder, f"{char}_{tail}"), char_resized)
            print(f"{crop_path}/{char}_{tail},coordinates:{x},{y},{w},{h}")
            


for i in plate_path.iterdir():
    img = cv.imread(i)
    top10 = segmentation(img)
    char_coors = crop_contours(top10)
    char_crop_save(char_coors,i)
    



