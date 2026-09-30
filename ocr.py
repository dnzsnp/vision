import os 
import matplotlib.pyplot as plt 
import numpy as np 
import cv2 as cv 
from pathlib import Path
import itertools


cropped_plates = Path("photos/cplates")
crop_path = Path("ocr/chars")

head,tail = os.path.split(cropped_plates)
count = 0 

def segmentationChar(img):
     img_grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
     blur = cv.GaussianBlur(img_grey,(5,5),3)
     thres_img = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
     kernel = np.ones((3,3), np.uint8)
     closed = cv.morphologyEx(thres_img, cv.MORPH_CLOSE, kernel, iterations=1)
     contours,hierarchy = cv.findContours(closed,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
     #draw_contours = cv.drawContours(img,contours,-1,(0,0,0),1)
     top10 = sorted(contours,key=lambda c:cv.contourArea(c,False), reverse=True)[:10]
     return top10 


def crop_contours(contours):
     char_coor=[]
     for i , c in enumerate(top10):
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
            head,tail = os.path.split(file_path)
            plt.ion()
            x,y,w,h = char_list[i]
            char_crop = img[y:y+h, x:x+w]
            padded_charcrop = cv.copyMakeBorder(char_crop,2,2,2,2,cv.BORDER_CONSTANT)
            char_resized = cv.resize(padded_charcrop,(32,32))
            plt.subplot(121)
            plt.title(file_path)
            plt.imshow(char_resized)
            plt.show(block=False)
            character = input("character?")
            
            if character == ",":
                  print(f"{crop_path}/{character}_{tail},unreadable!")
                  continue
            plt.close()
            folder = os.path.join(crop_path, character)
            if os.path.isdir(folder):
                   #buneamk#cv.imwrite(os.path.join(folder,1"", f"{character}_{tail}"), char_resized)
                  print("folder exists.")
            else:
                  os.mkdir(folder)      
            cv.imwrite(os.path.join(folder, f"{character}_{i}_{tail}"), char_resized)
            print(f"{crop_path}/{i}_{tail},coordinates:{x},{y},{w},{h}")


def char_save(char_list, file_path, img):
    for i, v in enumerate(char_list):
        x, y, w, h = char_list[i]
        char_crop = img[y:y+h, x:x+w]
        padded_charcrop = cv.copyMakeBorder(char_crop, 2, 2, 2, 2, cv.BORDER_CONSTANT)
        char_resized = cv.resize(padded_charcrop, (128, 128))

        cv.imshow("Character", char_resized)
        cv.waitKey(1)

        char = input("Character? (',' = unreadable, 'q' = quit): ").lower() # inputu lower yap 
        if char == "q":
            cv.destroyAllWindows()
            return
        if char == ",":
            continue

        folder = os.path.join(crop_path, char)
        os.makedirs(folder, exist_ok=True)
        filename = f"{char}_{i}_{file_path.name}"   # <- tail yerine file_path.name
        cv.imwrite(os.path.join(folder, filename), char_resized)
        print(f"{folder}/{filename}")
          







for i in itertools.islice(cropped_plates.iterdir(), None):
    img = cv.imread(i)
    top10 = segmentationChar(img)
    char_coors = crop_contours(top10)
    char_save(char_coors, i,img)

for i in crop_path.iterdir():
     if i.is_dir():
          count = sum(1 for f in i.iterdir() if f.is_file())
          print(f"{i.name}:{count}")

# sonda dosya isimlerini(karakterleri) içindeki dosya sayısıyla yaz 
     

