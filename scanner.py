#!/usr/bin/env python3
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Using a non-tuple sequence")
import sys
import subprocess
import threading
import traceback

LOFR = ''
FAST_VOA = ''
ISOLATION_FOREST = ''
SVMR = ''


def LOF():
    global LOFR
    LOFR = subprocess.check_output(["python3", "/home/mma/avserver/examples/LOF.py", sys.argv[1]], shell=False)
    LOFR = LOFR.decode()
    LOFR = LOFR.split("\n")[:-1]


def fast_voa():
    global FAST_VOA
    FAST_VOA = subprocess.check_output(["python3", "/home/mma/avserver/examples/fastVOA.py", sys.argv[1]], shell=False)
    FAST_VOA = FAST_VOA.decode()
    FAST_VOA = FAST_VOA.split("\n")[:-1]


def isolation_forest():
    global ISOLATION_FOREST
    ISOLATION_FOREST = subprocess.check_output(
        ["python3", "/home/mma/avserver/examples/IsolationForest.py", sys.argv[1]], shell=False)
    ISOLATION_FOREST = ISOLATION_FOREST.decode()
    ISOLATION_FOREST = ISOLATION_FOREST.split("\n")[:-1]


def SVM():
    global SVMR
    SVMR = subprocess.check_output(["python3", "/home/mma/avserver/examples/SVM.py", sys.argv[1]], shell=False)
    SVMR = SVMR.decode()
    SVMR = SVMR.split("\n")[:-1]


thread1 = threading.Thread(target=LOF, )
thread1.start()
thread2 = threading.Thread(target=fast_voa, )
thread2.start()
thread3 = threading.Thread(target=isolation_forest, )
thread3.start()
thread4 = threading.Thread(target=SVM, )
thread4.start()


for i in range(len(SVMR)):
    score = 0
    if float(SVMR[i]) > 0.90:
        score += 1
    if float(ISOLATION_FOREST[i]) > 0.90:
        score += 1
    if float(LOFR[i]) > 0.90:
        score += 1
    if float(FAST_VOA[i]) > 0.90:
        score += 1.5
    if score > 2.5:
        print("#2#" + str(i) + "#" + ISOLATION_FOREST[i] + "#" + FAST_VOA[i] +
              "#" + LOFR[i] + "#" + SVMR[i] + "#")
    elif score > 2:
        print("#1#" + str(i) + "#" + ISOLATION_FOREST[i] + "#" + FAST_VOA[i] +
              "#" + LOFR[i] + "#" + SVMR[i] + "#")

