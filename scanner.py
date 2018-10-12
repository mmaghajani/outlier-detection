#!/usr/bin/env python3
import warnings

from sklearn.metrics import roc_curve, roc_auc_score

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Using a non-tuple sequence")
import sys
import subprocess
import threading
from examples.modules import clustering as cls
import pandas as pd
from examples.modules import utils
import numpy as np
import traceback

LOFR = ''
FAST_VOA = ''
ISOLATION_FOREST = ''
SVMR = ''

ROOT = "/home/mma/avserver/"


def LOF():
    global LOFR
    LOFR = subprocess.check_output(["python3", ROOT + "examples/LOF.py", sys.argv[1]], shell=False)
    LOFR = LOFR.decode()
    LOFR = LOFR.replace(']', '').replace('[', '')
    LOFR = LOFR.split("\n")[:-1]


def fast_voa():
    global FAST_VOA
    FAST_VOA = subprocess.check_output(["python3", ROOT + "examples/fastVOA.py", sys.argv[1]], shell=False)
    FAST_VOA = FAST_VOA.decode()
    FAST_VOA = FAST_VOA.replace(']', '').replace('[', '')
    FAST_VOA = FAST_VOA.split("\n")[:-1]


def isolation_forest():
    global ISOLATION_FOREST
    ISOLATION_FOREST = subprocess.check_output(
        ["python3", ROOT + "examples/IsolationForest.py", sys.argv[1]], shell=False)
    ISOLATION_FOREST = ISOLATION_FOREST.decode()
    ISOLATION_FOREST = ISOLATION_FOREST.replace(']', '').replace('[', '')
    ISOLATION_FOREST = ISOLATION_FOREST.split("\n")[:-1]


def SVM():
    global SVMR
    SVMR = subprocess.check_output(["python3", ROOT + "examples/SVM.py", sys.argv[1]], shell=False)
    SVMR = SVMR.decode()
    SVMR = SVMR.replace(']', '').replace('[', '')
    SVMR = SVMR.split("\n")[:-1]


thread1 = threading.Thread(target=LOF, )
thread1.start()
thread1.join()
thread2 = threading.Thread(target=fast_voa, )
thread2.start()
thread2.join()
thread3 = threading.Thread(target=isolation_forest, )
thread3.start()
thread3.join()
thread4 = threading.Thread(target=SVM, )
thread4.start()
thread4.join()

for i in range(len(SVMR)):
    score = 0
    if float(SVMR[i]) > 20000:
        score += 1
    if float(ISOLATION_FOREST[i]) > 0:
        score += 1.5
    if float(LOFR[i]) > 2:
        score += 1
    if float(FAST_VOA[i]) > 0.5:
        score += 1
    if score >= 2.5:
        print("#2#" + str(i) + "#" + ISOLATION_FOREST[i] + "#" + FAST_VOA[i] +
              "#" + LOFR[i] + "#" + SVMR[i] + "#")
    elif score > 1.5:
        print("#1#" + str(i) + "#" + ISOLATION_FOREST[i] + "#" + FAST_VOA[i] +
              "#" + LOFR[i] + "#" + SVMR[i] + "#")

