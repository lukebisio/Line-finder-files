#This file populates a SQLite database with information about various lines
#Authors: Larry Donahue, Malachy Bloom, Michael Yang, Vincent He, Daniel Nykamp

import numpy as np
import os
import math as m
import time as t
from flask import Flask, render_template, request, Response, send_file
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///line_finder.db'
db = SQLAlchemy(app)

verbose = False
rootdir = os.getcwd()
threshold = 0.00 #Lines with coherences below this threshold will be ignored
chunksize = 50 #Width of chunks pushed to database

#The database itself, this is the class containing information about each row
class weekly(db.Model): #Weekly coherences
    id = db.Column(db.Integer, primary_key=True) #Unique ID for each line
    run = db.Column(db.String(10), index=True) #Run for each line
    obs = db.Column(db.String(10)) #Observatory for each line
    time = db.Column(db.Integer, index=True) #Epoch timestamp for each line, that can be converted into a date (e.g. 1483228800, which can become 2017/01/01)
    channel = db.Column(db.String(50)) #Channel for each line (e.g. L1_PEM-CS_MAG_LVEA_VERTEX_Z)
    freq = db.Column(db.Float, index=True) #Frequency for each line
    coh = db.Column(db.Float, index=True) #Coherence for each line

    def __repr__(self):
        return f"W,{self.run},{self.obs},{self.time},{self.channel},{self.freq},{self.coh}"

class monthly(db.Model): #Monthly coherences. Identical to weekly coherences, with the important distinction of range that necessitates a new table/database model.
    id = db.Column(db.Integer, primary_key=True) #Unique ID for each line
    run = db.Column(db.String(10), index=True) #Run for each line
    obs = db.Column(db.String(10)) #Observatory for each line
    time = db.Column(db.Integer, index=True) #Epoch timestamp for each line, that can be converted into a date (e.g. 1483228800, which can become 2017/01/01)
    channel = db.Column(db.String(50)) #Channel for each line (e.g. L1_PEM-CS_MAG_LVEA_VERTEX_Z)
    freq = db.Column(db.Float, index=True) #Frequency for each line
    coh = db.Column(db.Float, index=True) #Coherence for each line

    def __repr__(self):
        return f"M,{self.run},{self.obs},{self.time},{self.channel},{self.freq},{self.coh}"


def print_verbose(string):
    if verbose:
        print(string)

# Skims files in data folder and stores lines in temporary chunk files
def skim():

    temp_num = 0
    sig_lines = []

    # Creates a file containing a small selection of line objects
    def make_temp():

        nonlocal temp_num, sig_lines

        print_verbose("\nMoving currently stored lines into a temp file.")

        temp_num += 1
        tempname = "temp/chunk" + str(temp_num) + ".lft"
        tempfile = open(tempname, "w")
        tempfile.writelines([str(l) + "\n" for l in sig_lines])
        tempfile.close()

        sig_lines.clear()

        print_verbose("Temporary file " + tempname + " containing " + str(len(sig_lines)) + " lines written to temp directory.\n")

    file_list = []
    for subdir, _, files in os.walk(rootdir + "/data"):
        for file in files:
            if file.endswith('.txt'):
                filepath = os.path.join(subdir, file)
                file_list.append(filepath)

    os.mkdir("temp") #Creates directory. This directory should not exist at this point.

    print("Beginning file reading...")
    for file in file_list:
        numsiglines = len(sig_lines)

        # /home/daniel.nykamp/data/weekly/L1/fscans_2019_12_21_18_00_03_CST_Sat/L1_PEM-EX_MAG_VEA_FLOOR_Y_DQ/spec_1700.00_1800.00_1260403220_coherence_L1_PEM-EX_MAG_VEA_FLOOR_Y_DQ_and_L1_GDS-CALIB_STRAIN.txt
        path_split = file.split('/')[3:]
        date_split = path_split[3].split('_')

        datatype = path_split[1]
        obs = path_split[2]
        week = date_split[1] + "-" + date_split[2] + "-" + date_split[3]
        channel = path_split[4]
        time = int(t.mktime(t.strptime(week, '%Y-%m-%d')))
        run = run_by_date(date_split[1], date_split[2])

        print_verbose("Currently working with file: " + path_split[5] + " for channel " + channel + " in week " + week + "...")

        data = open(file, "r")
        for line in data:
            currline = line.split(' ') #Splits read line into frequency (index 0) and associated coherence (index 1)
            if currline == ['']:
                break

            (mantissa, exponent) = currline[0].split('e')
            freq = float(mantissa) * (10 ** float(exponent))
            freq = np.round(freq, 6)

            coh = float(currline[1]) #yStoring the coherence is... simpler.

            if threshold < coh and coh < 1:
                sig_lines.append((datatype, freq, coh, channel, time, obs, run))

        print_verbose("Moving to next file. Found " + str(len(sig_lines) - numsiglines) + " significant lines in file")

        if len(sig_lines) > chunksize:
            make_temp()

    if len(sig_lines) > 0:
        make_temp()



# Find out which line belongs to which run, based on time.
def run_by_date(year, month):
    if int(year) == 2019 and int(month) < 4:
        return 'ER14' #The experimental run before O3 began took place between 2019-03-01 and 2019-04-01
    elif int(year) == 2019 and int(month) < 11:
        return 'O3A' #The first part of the O3 observing run took place between ER14 and O3B
    else:
        return 'O3B' #The second part of the O3 observing run took place between 2019-11-01 and 2020-03-28


# A problem that arose as we neared production: We can't commit large (>3000000, at least) chunks to the database all at once.
# To solve this, we cut the speed of the process (which is fine, because this program runs once) by populating the database in small chunks.
# These chunks are stored in a 'temp' directory, and are read and stored one by one.
def populate():

    totalcls = 0 #A counting variable for total lines committed

    #An array of numbers which dictates the order of the temp files that get pushed to database.
    chunk_indices = [int(f.strip("chunk.lft")) for f in os.listdir(rootdir + "/temp")]
    chunk_indices.sort() #Sorts list numerically for easier time in for loop

    chunk_num = len(chunk_indices)
    print(str(chunk_num) + " chunk(s) will be used.")

    for chunk in chunk_indices: #For the number of chunks necessary...
        tempname = "temp/chunk" + str(chunk) + ".lft"

        tempfile = open(tempname, "r") #Reads chunk file in temp, stores lines in list
        sig_lines = [eval(l) for l in tempfile.readlines()]
        tempfile.close() #Closes file

        if len(sig_lines) == 0: #If there's no lines to look at, commit what we have
            print("End of line list reached.")
            os.remove(tempname)
            break

        totalcls += len(sig_lines)

        while len(sig_lines) > 0:
            datatype, freq, coh, channel, time, obs, run = sig_lines.pop(0) #Pop line off of list

            if datatype == "weekly":
                line = weekly(freq=freq, coh=coh, time=time, run=run, channel=channel, obs=obs)
            else:
                line = monthly(freq=freq, coh=coh, time=time, run=run, channel=channel, obs=obs)

            db.session.add(line) #Add line to DB session

        # When a chunk is fully added, push the commit and delete the temp
        db.session.commit()
        os.remove(tempname)
        print_verbose(str(chunk) + "/" + str(chunk_num) + ", " + str(totalcls) + " lines so far.")

    return totalcls

def prompt_user(question, dictionary):
    while True:
        user_input = input(question).lower()
        if user_input in dictionary:
                return dictionary[user_input]
        print("Invalid character.")

if __name__ == "__main__":
    print("Running...")

    #Prompt user to see whether init should run with extra verbosity or not
    verbose = prompt_user("Run init.py with extra verbosity? (Y/N): ", dict(y = True, n = False))

    save_exists = "temp" in os.listdir(rootdir)
    db_exists =  "line_finder.db" in os.listdir(rootdir)

    #Generate line list to commit to database. If a temp directory exists, pull sig lines from there as built into populate(). If not, read data directory.
    if save_exists:
        print("Resuming partial commitment.")
    else:
        skim()
        print("Beginning commitment of significant lines to the database.")

    # If the database doesn't exist, create the tables. Otherwise, don't drop what we have.
    with app.app_context():
        if not db_exists:
            db.drop_all()
            db.create_all()

        totallines = populate()

    print("Finished commitment of " + str(totallines) + " lines to the database.")

    #Check if temp directory exists again for removal.
    if "temp" in os.listdir(rootdir):
        os.rmdir("temp") #Discards temp directory
