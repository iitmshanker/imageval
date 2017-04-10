# imageval
#!/usr/bin/env python3

# usage:   python captcha_recognizer.py --image captcha.jpg
# Please cite the work :-)

import os
import logging
import argparse
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

# print the string after removing special characters
# present due to noise
def print_without_special( string):
	l = len(string)
	out = ''
	for i in range (l):
		c = ord(string[i])
		if ( c > 47 and c < 58) or ( c > 64 and c < 91) or ( c > 96 and c < 123):
			out =  out + chr(c)
	return out
	
# analyze all the results from various methods and find one final out
def process_results(gray_text,dil_text,dil2_text, dil3_text, seg_text, dil4_text, ero_text, cont_text ):
    if( dil3_text == seg_text):  #mostly we are correct 
        return dil3_text
    else:   # when two are not equal, one has to get preference
        if len(dil3_text) ==5 and len(seg_text) ==5:    # if not then we see which one has lenght 5
            # check if one of them is more similar to other results
            score_dil3 = common_presence(dil3_text, gray_text,dil_text,dil2_text, dil4_text, ero_text, cont_text)
            score_seg = common_presence(seg_text, gray_text,dil_text,dil2_text, dil4_text, ero_text, cont_text)
            if score_dil3 > 0 or score_seg >0:
                if score_dil3 > score_seg:
                    return dil3_text
                else:
                    return seg_text
            # check which letter is different and find the weight for that letter
            score_dil3 = 0
            score_seg = 0
            for i in range (5):
                if dil3_text[i] != seg_text[i]:
                    score = common_presence(dil3_text[i], gray_text,dil_text,dil2_text, dil4_text, ero_text, cont_text)
                    score_dil3 = score_dil3 + score
                    score = common_presence(seg_text[i], gray_text,dil_text,dil2_text, dil4_text, ero_text, cont_text)
                    score_seg = score_seg + score
            # we have scores for both whichever is better return it
            if score_dil3 > 0 or score_seg >0:
                if score_dil3 > score_seg:
                    return dil3_text
                else:
                    return seg_text
        else: # if lengths not equal to 5, return whichever is of length 5
            if len(dil3_text) ==5:
                return dil3_text
            if len(seg_text) == 5:
                return seg_text
        # even if we are here we shall see which one is of leggth 5
        other_result = explore_others(gray_text,dil_text,dil2_text, dil4_text, ero_text, cont_text) 
    #default
    if other_result == False:
        return dil3_text
    else:
        return other_result
# return the number of occurences of st in all the strings named texts
def common_presence(st, text1,text2,text3, text4,text5, text6):
    texts = [text1]+ [text2]+[text3] + [text4] + [text5] +[text6]
    score = 0
    for i in range(6):
        if ( st in texts[i]):
            score = score +1  
    return score

def explore_others(text1,text2,text3, text4,text5, text6):
    texts = [text1]+ [text2]+[text3] + [text4] + [text5] +[text6]
    best = -1
    best_score = 0
    for i in range (6):
        if len(texts[i]) == 5:
            score = common_presence(texts[i], texts[0],texts[1],texts[2],texts[3],texts[4],texts[5])
            if score > best_score:
                best_score = score
                best = i
                #print(texts[i])
    if best > -1:
        return texts[best]
    return False
    
# parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

#read the image 
image = cv2.imread(args["image"])
# remove colored lines
h = image.shape[0]
w = image.shape[1]

for j in range (w):
	for i in range (h):
	    # find rgb values of each pixel
		r = image[i][j][0]
		g = image[i][j][1]
		b = image[i][j][2]
		# if the pixel is neither white nor black
		if (r< 70 and g < 70 and b< 70):   # it's black
			continue
		if (r> 240 and g> 240 and b > 240):  # it's white
			continue
		image[i][j][0] = 255
		image[i][j][1] = 255
		image[i][j][2] = 255
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Remove all the pixels 1/8 left and 1/8 right side
for j in range ( 0, int(w/8)):
	for i in range (h):
		gray[i][j] = 255
		gray[i][w-j-1] = 255
# remove all the pixels 1/6 above and 1/6 bottom side
for j in range (0, w):
	for i in range(0, int(h/6)):
		#gray[i][j] = 255
		gray[h-i-1] = 255
# Remove thin horizontal lines from gray
for j in range (w):
	for i in range (2, h-3):
		if gray[i][j] < 100:  # pixel is dark
			if gray[i-1][j] < 100: # it has dark pixel just above
				continue
			if gray[i+1][j] >100 or gray[i+2][j] > 100 : # pixel down white 
				gray[i][j] = 230
cv2.imwrite('gray.jpg', cv2.resize(gray, None, fx =2, fy =2))
gray_copy = gray.copy()

# To get dilation
_,thresh = cv2.threshold(gray, 150,250,cv2.THRESH_BINARY_INV)   # experiment here
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(4,1))
dilated = cv2.dilate(thresh,kernel,iterations = 1)

# Remove thin horizontal lines from dilated
for j in range (w):
	for i in range (2, h-3):
		if dilated[i][j] > 150:  # pixel is white
			if dilated[i-1][j] > 150: # it has white pixel just above
				continue
			if dilated[i+1][j] < 50 or dilated[i+2][j] < 50 : # pixel down are black 
				dilated[i][j] = 10
# removal is done
cv2.imwrite('dilated.jpg', dilated)

# convert dilated to normal  black and white
for j in range ( 0, w):
		for i in range (h):
			if dilated[i][j] > 200:
				dilated[i][j] = 0
			elif dilated[i][j] < 50:
				dilated[i][j] = 255
cv2.imwrite('dilated2.jpg', dilated)
# dilate the new gray
_,thresh = cv2.threshold(gray, 150,250,cv2.THRESH_BINARY_INV)   # experiment here
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,1))
kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
dilated3 = cv2.dilate(thresh,kernel,iterations = 1)
# convert dilated to normal  black and white
for j in range ( 0, w):
		for i in range (h):
			if dilated3[i][j] > 200:
				dilated3[i][j] = 0
			elif dilated3[i][j] < 50:
				dilated3[i][j] = 255
#dilated3 = cv2.resize(dilated3, None, fx= 2, fy= 2)
cv2.imwrite('dilated3.jpg', dilated3)

# Perform OCR 
im_ocr = Image.open('gray.jpg')
ocr_text = pytesseract.image_to_string(im_ocr)
gray_text = print_without_special(ocr_text)
#print ("gray: ", gray_text )

im_ocr = Image.open('dilated.jpg')
ocr_text = pytesseract.image_to_string(im_ocr)
dil_text = print_without_special(ocr_text)
#print("dialted: ", dil_text)

im_ocr = Image.open('dilated2.jpg')
ocr_text = pytesseract.image_to_string(im_ocr)
dil2_text = print_without_special(ocr_text)
#print("dilated2: ", dil2_text)

im_ocr = Image.open('dilated3.jpg')
ocr_text = pytesseract.image_to_string(im_ocr,config = '-psm 7')
dil3_text =  print_without_special(ocr_text)
#print("dilated3: ", dil3_text )

# Variation in dilated3 using 5 parts
d3_copy = dilated3.copy()
left = 0   # first part starts here
partition = [56, 84, 112, 140, 168]
limit = 5
seg_text = ''
for k in range (5):
    copy = d3_copy.copy()
    best = h
    right = partition[k]-limit
    for j in range(partition[k] -limit , partition[k]+ limit):
        count_black = 0
        for i in range (h):
            if copy[i][j] < 50:
                count_black = count_black +1
        if count_black < best:
            best = count_black
            right = j
    # now we have left and right get the copy
    for j in range(0, left):
        for i in range(h):
            copy[i][j] = 250
    for j in range(right, w):
        for i in range(h):
            copy[i][j] = 250
             
    #print the results
    cv2.imwrite("part.jpg", copy)
    im_ocr = Image.open('part.jpg')
    ocr_text = pytesseract.image_to_string(im_ocr, config = '-psm 1000')
    ch = print_without_special(ocr_text)
    #print("d3_part: ", ch )
    seg_text = seg_text + ch
    # update left 
    left = right
#print("Segmented: ", seg_text)
# extra dilation
dilated4 = cv2.dilate(dilated3,kernel2,iterations = 1)
cv2.imwrite("dilated4.jpg", dilated4)
im_ocr = Image.open('dilated4.jpg')
ocr_text = pytesseract.image_to_string(im_ocr)
dil4_text =  print_without_special(ocr_text)
#print("dilated4: ",dil4_text)

# erosion result
eroded = cv2.erode(dilated, kernel2, iterations =1)
cv2.imwrite('eroded.jpg', eroded)

im_ocr = Image.open('eroded.jpg')
ocr_text = pytesseract.image_to_string(im_ocr)
ero_text = print_without_special(ocr_text)
#print("eroded: ",ero_text)

# contour method
gray_copy = cv2.GaussianBlur(gray_copy, (5,5), 0)
gray_copy = cv2.Canny(gray_copy, 2, 10)

im2, cnts, hierarchy = cv2.findContours(gray_copy, cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse=True)
for screenCnt in cnts:
	if cv2.contourArea(screenCnt) > 10:
		cv2.drawContours(gray_copy, [screenCnt], -1, (255, 255, 255), 2)

cv2.imwrite("contour.jpg", gray_copy)
im_ocr = Image.open('contour.jpg')
ocr_text = pytesseract.image_to_string(im_ocr)
cont_text = print_without_special(ocr_text)
#print ("contour: ", cont_text)

output = process_results(gray_text,dil_text,dil2_text, dil3_text, seg_text, dil4_text, ero_text, cont_text)
print( "Here is the captcha result: ", output)

