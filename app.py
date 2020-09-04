from __future__ import division, print_function
import os
import shutil
import numpy as np
import zipfile
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.models import load_model
from flask import Flask, send_file, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
model_Path = 'models/'
# Load your trained model
CGrange0 = load_model(model_Path + '\cgrange0fulldata.h5')
CGrange1 = load_model(model_Path + '\cgrange1fulldata.h5')
CGrange2 = load_model(model_Path + '\cgrange2fulldata.h5')
CGrange3 = load_model(model_Path + '\cgrange3fulldata.h5')
CGrange4 = load_model(model_Path + '\cgrange4fulldata.h5')
CGrange5 = load_model(model_Path + '\cgrange5fulldata.h5')
CGrange6 = load_model(model_Path + '\cgrange6fulldata.h5')
CGrange7 = load_model(model_Path + '\cgrange7fulldata.h5')
CGrange8 = load_model(model_Path + '\cgrange8fulldata.h5')
CGrange9 = load_model(model_Path + '\cgrange9fulldata.h5')
# Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')
def get_candidategenes(orfinput, cgrange):
    if cgrange <= 0.3657:
        class_pred_full = CGrange0.predict_classes(orfinput)
    if cgrange > 0.3657 and cgrange <= 0.4157:
        class_pred_full = CGrange1.predict_classes(orfinput)
    if cgrange > 0.4157 and cgrange <= 0.46:
        class_pred_full = CGrange2.predict_classes(orfinput)
    if cgrange > 0.46 and cgrange <= 0.5014:
        class_pred_full = CGrange3.predict_classes(orfinput)
    if cgrange > 0.5014 and cgrange <= 0.5428:
        class_pred_full = CGrange4.predict_classes(orfinput)
    if cgrange > 0.5428 and cgrange <= 0.5814:
        class_pred_full = CGrange5.predict_classes(orfinput)
    if cgrange > 0.5814 and cgrange <= 0.6185:
        class_pred_full = CGrange6.predict_classes(orfinput)
    if cgrange > 0.6185 and cgrange <= 0.65:
        class_pred_full = CGrange7.predict_classes(orfinput)
    if cgrange > 0.65 and cgrange <= 0.6828:
        class_pred_full = CGrange8.predict_classes(orfinput)
    if cgrange > 0.6828 and cgrange <= 1:
        class_pred_full = CGrange9.predict_classes(orfinput)
    return class_pred_full

#checking the dna

#onehot encoding
def onehote(sequence):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq2 = [mapping[i] for i in sequence]
    return np.eye(4)[seq2]
# padding

def queryPosition(strt,lenth):
   return strt*3,strt*3+lenth*3
def strand(x):
    return "+" if x < 3 else "-"
translateaminoacides = {    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
                            'TGC':'C', 'TGT':'C', 'GAC':'D', 'GAT':'D',
                            'GAA':'E', 'GAG':'E', 'TTC':'F', 'TTT':'F',
                            'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
                            'CAC':'H', 'CAT':'H', 'ATA':'I', 'ATC':'I',
                            'ATT':'I', 'AAA':'K', 'AAG':'K', 'CTT':'L',
                            'TTA':'L', 'TTG':'L',
                            'CTA':'L', 'CTC':'L', 'CTG':'L', 'ATG':'M',
                            'AAC':'N', 'AAT':'N', 'CCA':'P', 'CCC':'P',
                            'CCG':'P', 'CCT':'P', 'CAA':'Q', 'CAG':'Q',
                            'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
                            'AGA':'R', 'AGG':'R', 'AGC':'S', 'AGT':'S',
                            'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
                            'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
                            'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
                            'TGG':'W', 'TAC':'Y', 'TAT':'Y', 'TAA':'*',
                            'TAG':'*','TGA':'*'}

def padding(onehotenc):
    # make the lenght 700 so the padding becomes 700
    addition = [[0, 0, 0, 0] for i in range(0, 700 - len(onehotenc[0]), 1)]
    onehotenc[0] = np.concatenate([onehotenc[0], addition])
    return np.stack(tf.keras.preprocessing.sequence.pad_sequences(onehotenc, padding="post"))

def model_predict(file_path, outputfile, minOrfLenght, unresolvedCodon, filetype):
     if filetype.lower() == 'nucl':
        nuclOutName = outputfile + '_nucl.fasta'
        coordOutName = outputfile + '_CNN-MGP_coord.coords'
     if filetype.lower() == 'prot':
        protOutName = outputfile + '_prot.fasta'
        coordOutName = outputfile + '_CNN-MGP_coord.coords'
     else:
        protOutName = outputfile + '_prot.fasta'
        nuclOutName = outputfile + '_nucl.fasta'
        coordOutName = outputfile + '_CNN-MGP_coord.coords'
     startCodons = []
     stopCodons = ["TAA", "TAG", "TGA"]
     if unresolvedCodon == 1:
         startCodons = ["ATG", "TTG", "GTG", "CTG"]
     else:
         startCodons = ["ATG"]
     def check_Dna(orf):
         index = 0
         dna_checked = False
         codonA, codonT, codonC, codonG = 0, 0, 0, 0
         if len(orf) > minOrfLenght:
             for ind, codon in enumerate(orf):
                 codonA += codon.count('A')
                 codonT += codon.count('T')
                 codonC += codon.count('C')
                 codonG += codon.count('G')
                 if codon.count('A') + codon.count('C') + codon.count('T') + codon.count('G') == 3:
                     dna_checked = True
                     index = ind + 1
                 elif len(orf[:ind]) > minOrfLenght:
                     dna_checked = True
                     index = ind
                     break
                 else:
                     break
         else:
             dna_checked = False
         if codonA > 1 and codonT > 1 and codonC > 1 and codonG > 1:
             return dna_checked, orf[:index]
         else:
             return False, []
     def create_OutputFiles(predictedGene):
         if filetype.lower() == 'nucl':
             nucORFsfile = open("output\\"+nuclOutName, 'a+')
             nucORFsfile.write("".join(predictedGene) + "\n")
         elif filetype.lower() == 'prot':
             protORFsfile = open("output\\"+protOutName, 'a+')
             for codon in range(0, len(predictedGene), 3):
                 protORFsfile.write(translateaminoacides[predictedGene[codon:codon + 3]])
             protORFsfile.write("\n")
         else:
             protORFsfile = open("output\\"+protOutName, 'a+')
             nucORFsfile = open("output\\"+nuclOutName, 'a+')
             for codon in range(0, len(predictedGene), 3):
                 protORFsfile.write(translateaminoacides[predictedGene[codon:codon + 3]])
             protORFsfile.write("\n")
             nucORFsfile.write("".join(predictedGene) + "\n")

     orfscoords = open("output\\"+coordOutName, 'a+')
     sequences = []
     descr = None
     # here is the path of multifalsta file
     with open(file_path) as file:
         line = file.readline()[:-1]  # always trim newline
         while line:
             if line[0] == '>':
                 if descr:  # any sequence found yet?
                     sequences.append((descr, seq))
                 descr = str(line[1:].split('>'))
                 seq = ''  # start a new sequence
             else:
                 seq += line
             line = file.readline()[:-1]
         sequences.append((descr, seq))
     for index, value in enumerate(sequences):  # extracting all orf in each fragment then predicting  coding from non-coding orfs
        Frames = []
        dna = value[1]  # extract the fragment
        description = value[0]
        Frames = []
        reversedna = []
        listOfOrfs = []
        listOfOrfID = []
        # create the positive frames
        # split the frames into codons for better performance
        Frames.append([dna[i:i + 3] for i in range(0, len(dna), 3)])
        Frames.append([dna[i:i + 3] for i in range(1, len(dna), 3)])
        Frames.append([dna[i:i + 3] for i in range(2, len(dna), 3)])
        # reverse compliment of the fragment
        reverse = {"A": "T", "C": "G", "T": "A", "G": "C"}
        for i in range(len(dna)):
            reversedna.append(reverse[dna[-i - 1]]) if dna[-i - 1] in reverse.keys() else reversedna.append(
                dna[-i - 1])  # if any contamination found we keep it for further more check
        reversedna = ''.join(reversedna)
        # create the negative frames
        Frames.append([reversedna[i:i + 3] for i in range(0, len(reversedna), 3)])
        Frames.append([reversedna[i:i + 3] for i in range(1, len(reversedna), 3)])
        Frames.append([reversedna[i:i + 3] for i in range(2, len(reversedna), 3)])
        #  our models predict genes for each cgrange with a different model
        CG_count = (dna.count('C') + dna.count('G')) / len(dna)


        def extract_IncompleteORFS_From_TheStart(stran, frame):
            # get the first tree indixes of stop codons
            lenth = len(
                frame)  # if no stop or start codon found we take the lenght of the frame(only start orfs, only stop orfs or orfs missing both start and stop)
            TAG = frame.index("TAG") if "TAG" in frame else lenth
            TAA = frame.index("TAA") if "TAA" in frame else lenth
            TGA = frame.index("TGA") if "TGA" in frame else lenth
            # get the first four indixes of start codons
            ATG = frame.index("ATG") if "ATG" in frame else lenth
            TTG = frame.index("TTG") if "TTG" in frame else lenth
            GTG = frame.index("GTG") if "GTG" in frame else lenth
            CTG = frame.index("CTG") if "CTG" in frame else lenth
            if unresolvedCodon == 1:
                minStart = min(ATG, TTG, GTG, CTG)
            else:
                minStart = ATG
            if min(TAG, TAA, TGA) < minStart:  # the first codon found is stop : ------------stop--
                goodDna, orf = check_Dna(frame[:min(TAG, TAA, TGA) + 1])  # check the orf lenght and contamination
                if goodDna:
                    listOfOrfs.append(''.join(orf))
                    qstart, qend = queryPosition(0, min(TAG, TAA, TGA))
                    listOfOrfID.append([index * 2, strand(stran), qstart, qend, stran, "I"])
            elif min(TAG, TAA, TGA) == lenth and minStart < lenth:  # stop codon not found  but the start codons is, we take the first start codon :---start-----------------------
                goodDna, orf = check_Dna(frame[minStart:])  # check the orf lenght and contamination
                if goodDna:
                    listOfOrfs.append(''.join(orf))
                    qstart, qend = queryPosition(minStart, len(orf))
                    listOfOrfID.append([index * 2, strand(stran), 1, qstart, qend, stran, "I"])
            elif min(TAG, TAA, TGA) == lenth and minStart== lenth:  # neather stop or start codon found: ------------------------
                goodDna, orf = check_Dna(check_Dna(frame[:]))
                if goodDna:
                    listOfOrfs.append(''.join(orf))
                    qstart, qend = queryPosition(0, len(orf))
                    listOfOrfID.append([index * 2, strand(stran), 1, qstart, qend, stran, "I"])


        def extract_IncompleteORFS_From_TheEnd(stran, frame):
            lenth = len(frame)
            # searching for the last stop codon
            TAG = list(reversed(frame)).index("TAG") if "TAG" in frame else 0
            TAA = list(reversed(frame)).index("TAA") if "TAA" in frame else 0
            TGA = list(reversed(frame)).index("TGA") if "TGA" in frame else 0
            # getting the last stop codon index
            if min(TAG, TAA,TGA) != 0:  # if no codon stop found means eather only start or both codon missing, we already extacted that
                lastStopCodon = lenth - min(TAG, TAA, TGA) - 1  # getting the last index if stop codons
                # searching for anypossible  start codont after the last stop codon: --------------stop-----start-----------
                ATG = frame[lastStopCodon:].index("ATG") if "ATG" in frame[lastStopCodon:] else lenth
                TTG = frame[lastStopCodon:].index("TTG") if "TTG" in frame[lastStopCodon:] else lenth
                GTG = frame[lastStopCodon:].index("GTG") if "GTG" in frame[lastStopCodon:] else lenth
                CTG = frame[lastStopCodon:].index("CTG") if "CTG" in frame[lastStopCodon:] else lenth
                if unresolvedCodon == 1:
                    minStart = min(ATG, TTG, GTG, CTG)
                else:
                    minStart = ATG
                if minStart < len(frame):  # if no stop codon found  we exit the function otherwise we check the orf
                    newStart = lastStopCodon + minStart  # getting the new start position
                    goodDna, orf = check_Dna(frame[newStart:])  # checking the orf for lenght and contamination
                    if goodDna:
                        listOfOrfs.append(''.join(orf))
                        qstart, qend = queryPosition(newStart, len(orf))
                        listOfOrfID.append([index * 2, strand(stran), 1, qstart, qend, stran, "I"])


        # the extraction of the all orfs
        for i in range(0, len(Frames), 1):
            # delete the contamination if first codon or the last codon to boost the performance for the checkDna function
            if len(Frames[i][-1]) < 3 or "N" in Frames[i][-1]:
                del Frames[i][-1]
            if "N" in Frames[i][0]:
                del Frames[i][0]
            # extract all the incomplete orfs
            extract_IncompleteORFS_From_TheEnd(i, Frames[i])
            extract_IncompleteORFS_From_TheStart(i, Frames[i])
            # extract  all the complete  orfs
            start = 0
            while start < len(Frames[i]):
                # get the index of the first start
                if Frames[i][start] in startCodons:
                    # get the index of the first first stop
                    for stop in range(start + 1, len(Frames[i]), 1):
                        if Frames[i][stop] in stopCodons:
                            goodDna, orf = check_Dna(Frames[i][start:stop + 1])  # check lenght of the dna, contamination
                            if goodDna:
                                listOfOrfs.append(''.join(orf))
                                qstart, qend = queryPosition(start, len(orf))
                                listOfOrfID.append([index * 2, strand(i), 1, qstart, qend, i, "C"])
                                start = stop  # skip into the index after stop  then repeat the search
                            break
                start += 1
        # after the extraction of all orfs for this fragment; it's time for preprocessing the orfs,
        # the models only accepts orfs with lenght 700*4
        # first  all orfs should be one hot encoded  the (lenght*4)
        # second we performe padding (adding vectors of zeros (0,0,0,0) until the lenght 700 is satisfied
        if len(listOfOrfs) != 0:  # did we extract any ORFS
            onehotencodedorf = []
            # performe onehot encoding
            for seq in listOfOrfs:
                onehotencodedorf.append(onehote(seq))
            # padd the lenght to 700
            inputOrfs = padding(onehotencodedorf)
            # each cgrange has a specific model
            # feed the orfs into the CNN-MGP alongside the CG content of the fragment
            candidategenes = get_candidategenes(inputOrfs, CG_count)
            # add the predicted  genes into the outputfile
            for i in range(0, len(candidategenes), 1):
                if candidategenes[i] == 1:
                    create_OutputFiles(listOfOrfs[i])
                    orfscoords.write(">" + str(listOfOrfID[i]).replace('[', '').replace(']', '') + "\n")

@app.route('/')
def home():
    # Main page
    return render_template('index.html')
@app.route('/Submission.html')
def submit():
    return render_template('Submission.html')
@app.route('/index.html')
def index():
    return render_template('index.html')
@app.route('/Download.html')
def Download():
    return render_template('Download.html')
@app.route('/Results.html')
def results():
    return render_template('Results.html')
@app.route('/Help.html')
def Help():
    return render_template('Help.html')
@app.route('/Department.html')
def departemnt():
    return render_template('Department.html')
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':

        # Get the file from post request
        f = request.files['file']
        error = None
        try:
            orfLenght = int(request.form['minLenght'])
            if orfLenght<80 or orfLenght>700:
                error = 'input must be integer greater than 80 and less than 700, recommanded 80'
                return render_template('Submission.html', error=error)
        except ValueError:
            error = 'input must be integer '
            return render_template('Submission.html', error=error)
        outputfile="output"
        startType=int(request.form['ATGsatrt'])
        filetype=request.form['seqType']
        if not os.path.exists('output'):
            os.makedirs('output')
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        model_predict(file_path, outputfile, orfLenght, startType, filetype)
        zipf = zipfile.ZipFile('predicted.zip', 'w', zipfile.ZIP_DEFLATED)
        for root, dirs, files in os.walk('output/'):
            for file in files:
                zipf.write('output/' + file)
        zipf.close()
        print(
            'zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
        for (dirpath, dirnames, filenames) in os.walk("output"):
            print(filenames)
            for file in filenames:
                os.remove("output\\" + file)
        for (dirpath, dirnames, filenames) in os.walk("uploads"):
            for file in filenames:
                os.remove("uploads\\" + file)
        return send_file('predicted.zip',
                         mimetype='zip',
                         attachment_filename='predicted.zip',
                         as_attachment=True)
    os.remove("predicted.zip")


if __name__ == '__main__':
    app.run(debug=True)

