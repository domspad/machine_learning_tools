#!/usr/bin/python
#title			:daily_log.py
#description	:run at the end of every session, and will append which files have been changed to to the daily log
#author			:domspad
#date			:20140821
#version		:0.1
#usage			:python daily_log.py
#notes			:
#python_version	:2.6.6
#==============================================================================

import os, time

LOGFILE = 'daily.log'
NOMENTION = ['daily.log', 'ipynbs/nohup.out']

def line_prepender(filename,line):
    with open(filename,'r+') as f:
        content = f.read()
        f.seek(0,0)
        f.write(line.rstrip('\r\n') + '\n' + content)

with open(LOGFILE,'r') as logfile :
	lasttime = float(logfile.readline().strip())
	
FILES = []
for dirname, dirnames, filenames in os.walk('.'):
    for filename in filenames:
        fullname = os.path.join(dirname, filename)
        modtime = os.path.getmtime(fullname)
        if (modtime > lasttime) and ('checkpoint' not in fullname):
        	FILES.append((modtime, fullname[2:]))
FILES = sorted(FILES, key=lambda tup: tup[0])
for f in NOMENTION :
    if f in FILES :
        FILES.remove(f)
FILES.reverse()

output =  """{date}\n{cdate}\n\n{seshnotes}\n\n{modfiles}\n\n*******************************************************************************""".format(
			date=time.time(),
			cdate=time.ctime(),
            seshnotes=raw_input('what did you do...'),
            modfiles='\n'.join(f[1] for f in FILES))

line_prepender(LOGFILE,output)

