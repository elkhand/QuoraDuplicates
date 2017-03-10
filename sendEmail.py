#!/usr/bin/env python
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from smtplib import SMTP
import smtplib
import sys
import time
import datetime

#['elkhan.dadashov@gmail.com']
toEmail= sys.argv[1]
recipients = [toEmail]
emaillist = recipients
now = datetime.datetime.now()
while(True):
    msg = MIMEMultipart()
    msg['Subject'] = "CS224N Final Project Experiment "+str(now)
    fromEmail = 'cs224ndlnlp@gmail.com'
    msg['From'] = fromEmail
    msg['Reply-to'] = 'cs224ndlnlp@gmail.com'


    msg.preamble = 'Multipart massage.\n'
    part = MIMEText("Hi, please find the attached file")
    msg.attach(part)


    part = MIMEApplication(open(str(sys.argv[2]),"rb").read())
    part.add_header('Content-Disposition', 'attachment', filename=str(sys.argv[2]))
    msg.attach(part)


    server = smtplib.SMTP("smtp.gmail.com:587")
    server.ehlo()
    server.starttls()
    server.login(fromEmail, "azure123$")

    server.sendmail(msg['From'], emaillist , msg.as_string())
    time.sleep(600)# Sleep 10 minutes
