import numpy as np
from torch._C import device
import cv2
import argparse
import time

def get_box(res):
    points = res[0].copy()
    
    address_1   =   points[0][0]
    address_2   =   points[1][0]
    birthday    =   points[2][0]
    hometown_1  =   points[3][0]
    hometown_2  =   points[4][0]
    ids         =   points[5][0]
    name        =   points[6][0]
    nation      =   points[7][0]
    sex         =   points[8][0]
    
    info = {}

    ['address_line_1','address_line_2', 'birthday', 'hometown_line_1', 'hometown_line_2', 'id', 'name', 'nation', 'sex']


    info['address_line_1']  =   address_1
    info['address_line_2']  =   address_2
    info['birthday']        =   birthday
    info['hometown_line_1'] =   hometown_1
    info['hometown_line_2'] =   hometown_2
    info['id']              =   ids
    info['name']            =   name
    info['nation']          =   nation
    info['sex']             =   sex

    return info

