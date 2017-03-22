#!/usr/bin/env python
import subprocess
import sys
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from smtplib import SMTP
import smtplib
import sys
import time
import datetime
import re
t=600
toEmail= sys.argv[1]
recipients = [toEmail]
emaillist = recipients
now = datetime.datetime.now()
while(True):
    msg = MIMEMultipart()
    confInfo = ""
    if len(sys.argv)>3:
        confInfo = sys.argv[3]
        # this is for config and which machine
    msg['Subject'] = confInfo+" CS224N Final Project Experiment "+str(now)
    fromEmail = 'cs224ndlnlp@gmail.com'
    msg['From'] = fromEmail
    msg['Reply-to'] = 'cs224ndlnlp@gmail.com'

    fn=sys.argv[2]
    sWords = ["809/809","2429/2429","INFO:Epoch","100/101"]
    res=""
    with open(fn) as origin_file:
        for line in origin_file:
            line = line.encode('utf-8')
            for s in sWords:
                if s in line:
                    line = line[line.find(s):]
                    #print line
                    res+=line
    f1s=res
    print f1s

    msg.preamble = 'Multipart massage.\n'
    part = MIMEText(f1s)
    msg.attach(part)

    server = smtplib.SMTP("smtp.gmail.com:587")
    server.ehlo()
    server.starttls()
    server.login(fromEmail, "azure123$")

    server.sendmail(msg['From'], emaillist , msg.as_string())
    time.sleep(t)# Sleep t seconds
