import numpy as np
from numpy.core.einsumfunc import _parse_possible_contraction
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
    passport    =   points[9][0]
    
    THRESHOLD = 0.3

    info = {}

    info['address_line_1']  =   address_1   if address_1[-1] > THRESHOLD    else []
    info['address_line_2']  =   address_2   if address_2[-1] > THRESHOLD    else []
    info['birthday']        =   birthday    if birthday[-1] > THRESHOLD     else []
    info['hometown_line_1'] =   hometown_1  if hometown_1[-1] > THRESHOLD   else []
    info['hometown_line_2'] =   hometown_2  if hometown_2[-1] > THRESHOLD   else []
    info['id']              =   ids         if ids[-1] > THRESHOLD          else []
    info['name']            =   name        if name[-1] > THRESHOLD         else []
    info['nation']          =   nation      if nation[-1] > THRESHOLD       else []
    info['sex']             =   sex         if sex[-1] > THRESHOLD          else []
    info['passport']        =   passport    if passport[-1] > THRESHOLD     else []

    return info

