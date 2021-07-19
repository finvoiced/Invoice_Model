import cv2
import pytesseract
from pytesseract import Output
import numpy as np

pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def img_data(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_data(img, output_type='data.frame')
    text = text[text.conf != -1]
    lines = text.groupby(['page_num', 'par_num', 'line_num'])['text'] \
        .apply(lambda x: ' '.join(list(x))).tolist() #get everything on one line, that is why word_num not included
    confs = text.groupby(['page_num', 'par_num', 'line_num'])['conf'].mean().tolist() #taking average based on mean confidence of all words in a line. data frame returned is having confidence of each word in the line so we have to take a mean.

    line_conf = []
    k=0
    sentence = ''
    for i in range(len(lines)):
        if lines[i].strip():
            if round(confs[i], 3)>40:
                sentence= sentence +', ' + lines[i]
                line_conf.append([lines[i], round(confs[i], 3)])
                (x, y, w, h) = (text['left'].iloc[i], text['top'].iloc[i], text['width'].iloc[i], text['height'].iloc[i])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return img,line_conf,sentence
