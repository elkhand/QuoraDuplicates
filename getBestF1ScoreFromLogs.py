#!/usr/bin/python
# -*- coding: utf-8 -*-

# - * -coding: utf - 8 - * -

import sys
import time
import datetime
import re

pathToLogFile = sys.argv[1]

devStr = 'Evaluating on development data'
delimeter = 'acc/P/R/F1/loss:'
isDevResult = False

maxDevF1 = -1
maxAcc = -1
with open(pathToLogFile) as f:
    for line in f:
        line = line.encode('utf-8')
        if devStr in line:
            isDevResult = True
            continue
        if isDevResult:
            print line
            isDevResult = False
            values = line.split(delimeter)
            print 'Values: ', values
            values = values[1].strip()
            print 'Values: ', values
            values = values.split('/')
            f1 = values[3]
            acc = values[0]
            if f1 > maxDevF1:
                maxDevF1 = f1
            if acc > maxAcc:
                maxAcc = acc

print 'Max F1: ', maxDevF1
print 'Max Acc: ', maxAcc