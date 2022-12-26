from flask import Flask, render_template,request,flash,redirect,url_for,jsonify
import sqlite3
import string, re
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
from deep_translator import GoogleTranslator
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import TweetTokenizer
import nltk
import pandas as pd
from nltk.corpus import stopwords
from werkzeug.utils import secure_filename
import os
from werkzeug.utils import secure_filename
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import joblib
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences 
import numpy as np 
############################################### Cleansing Data ######################################################
 
def cleansing(data):
    # lower text
    data = data.lower()
    data = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', data)   
    #menghilangkan \...
    data = re.sub(r'\\[^\s]+',' ',data) 
    #menghilangkan #hashtag
    data = re.sub(r'#([^\s]+)', ' ', data)
    #menghilangkan @username
    data = re.sub(r'@[^\s]+',' ',data)
    #menghilangkan RT
    data = re.sub(r'rt ', ' ', data)
    #menghilangkan tanda baca
    data = re.sub(r'[^\w]|_',' ',data)
    #menghilangkan anga
    data = re.sub(r"\d+", " ", data)
    #menghilangkan spasi yang berlebih
    data = re.sub(r'[\s]+', ' ', data)
    
    
    # hapus punctuation
    remove = string.punctuation
    translator = str.maketrans(remove, ' '*len(remove))
    data = data.translate(translator)
    
    # remove ASCII dan unicode
    data = data.encode('ascii', 'ignore').decode('utf-8')
    data = re.sub(r'[^\x00-\x7f]',r'', data)
    
    # remove newline
    data = data.replace('\n', ' ')
    
    return data

############################################### Translate Text ###################################################### 

# def translate(data): 
#     data = ''.join(data).split()
#     for index, value in enumerate(data):
#         if value != '':
#             data[index] = GoogleTranslator(source='auto', target='id').translate(value) 

#     data = " ".join(data)
#     return data

############################################### Stemming Text ###################################################### 
# Buat Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(data):
    data = stemmer.stem(data)
    return data

############################################### Tokenisasi Text ###################################################### 
Tokenizing = TweetTokenizer()
    
def tokenisasi(data):
    data = Tokenizing.tokenize(data)
    return data

############################################### Stopwords Text ######################################################
# get stopword indonesia
list_stopwords = stopwords.words('indonesian')

list_stopwords.extend(['gue', 'gua', 'w', 'we', 'gw', 'ga', 'aja', 'saja','kaya', 'kayak', 'yg', 'yang', 'aku', 'wkwk', 'jg', 'jadi',
                 'si', 'di', 'kali', 'ye', 'lo', 'loe', 'trs', 'jd', 'lg', 'lagi','ama', 'sama', 'lah', 'kek', 'gak', 'gada', 'wkwkwkw', 'wkwkwwk',
                 'gaada', 'gw', 'gg', 'aduh', 'ah', 'iya', 'cuman', 'cuma', 'tp', 'tapi', 'aja', 'kalau', 'yu', 'wkakak', 'wah','wahhh',
                 'ka', 'kak', 'dah', 'deh', 'krn', 'dek', 'kyk', 'mau', 'plis', 'sama', 'yes', 'bs', 'ye','lha','loh','lu','kian',
                 'ni', 'in', 'ini', 'gk', 'ada', 'gak', 'mas', 'd', 'elah', 'aj', 'za', 'mo', 'kl', 'mah', 'nya', 'to', 'ha','siss',
                 'ak', 'drpd', 'huf', 'pas', 'sama', 'situ', 'yah', 'klo', 'u', 'i', 'nii', 'tpi', 'ku', 'ato', 'wkwkw', 'hadeh',
                 'dong', 'doang', 'apa', 'ada', 'masa', 'hahaha', 'tadi', 'gajadi', 'sini', 'sih', 'mau', 'pake', 'dulu','gin',
                 'gatau', 'utk', 'udah', 'dehh', 'gw', 'kalo', 'ingin', 'terus', 'bukan', 'punya', 'seperti', 'egk', 'buat', 
                 'akan', 'maka', 'gitu', 'punya', 'di', 'gabisa', 'nda', 'yak', 'dr', 'dari', 'tb', 'selalu', 'tau', 'yg', 
                 'yaa', 'lg', 'lagi', 'ini', 'eh', 'ma', 'malah', 'jangan', 'oke', 'kalau', 'yh', 'walaupun', 'wlopun', 'neh',
                 'teh', 'sm', 'sdh', 'ngeh', 'aq', 'atau', 'itu', 'dia', 'tadi', 'belum', 'buat', 'apasih', 'bgt', 'malah', 'ya',
                 'tuh', 'sudah', 'ken', 'segini', 'kak', 'bagaimana', 'biar', 'sudah', 'tuh', 'memang', 'bagaimana', 'enggak', 'nggak',
                  'kk', 'kaak', 'kaaa', 'kakk', 'kakkkkk', 'ngga', 'gk', 'gbs', 'gak', 'gda', 'engga', 'gakk','tidak', 'nih','eh', 'btw',
                  'ngk', 'gx', 'segini','dll','asa','via','rang','hahahahahahah','woy','sih','kes','seh','n'])

txt_stopword = pd.read_csv(r"C:/Users/ffadlurr/Documents/Binar Platinum/Platinum_Challenge/Group/Web-Flask-Sentiment-using-model-LSTM-Neural-Network-main/library/stopwords.txt", names= ["stopwords"], header = None)

# convert stopword string to list & append additional stopword
list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))

# convert list to dictionary
list_stopwords = set(list_stopwords)

# remove stopword pada list token
def stopword(data):
    return [word for word in data if word not in list_stopwords]

############################################### Abusive Text ###########################################################

# csv_data = pd.read_csv("D:/O2122121 Adms/Kuliah/Project Flask/Sentiment Analysis Using Model LSTM& Neural Network/library/abusive.csv")

# list = csv_data['ABUSIVE'].astype(str).tolist()
# list_abusive = set(list)
# def abusive(data):
#     return [word for word in data if word not in list_abusive]

############################################### Normalisasi Text ########################################################

normalisasi = pd.read_csv(r"C:/Users/ffadlurr/Documents/Binar Platinum/Platinum_Challenge/Group/Web-Flask-Sentiment-using-model-LSTM-Neural-Network-main/library/kamusalay.csv",encoding='utf-8')
normalisasi_dict = {}

for index, row in normalisasi.iterrows():
    if row[0] not in normalisasi_dict:
        normalisasi_dict[row[0]] = row[1]

def normalisasi(data):
    return [normalisasi_dict[term] if term in normalisasi_dict else term for term in data]

############################################### Join String Text ######################################################

def joinstring(data): 
    delimiter = ' '
    text = delimiter.join(data)
    return text

############################################### Combine All Function ###################################################

def preprocessing(text):
    hasil = joinstring(normalisasi(stopword(tokenisasi(stemming(cleansing(text))))))
    return hasil

############################################### End Function Preprocessing #############################################

def find_sentiment(review):
    seq = Tokenizer.texts_to_sequences(review)
    padded = pad_sequences(seq, maxlen=128)
    pred = ModelLSTM.predict(padded)
    label = ['Negative','Neutral','Positive']
    return label[np.argmax(pred)]

############################################### Function Upload Model ##################################################

app = Flask(__name__)
app.secret_key="123"

# enable debugging mode
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

# Allow Extension
ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.json_encoder = LazyJSONEncoder 

swagger_template = dict(
    info = {
        'title': LazyString(lambda:'Sentiment Analyst Used Model NN & LSTM'),
        'version': LazyString(lambda:'1.0.0'),
        'description': LazyString(lambda:'Based Input Text')
        }, host = LazyString(lambda: request.host)
    )

swagger_config = {
        "headers":[],
        "specs":[
            {
            "endpoint":'docs',
            "route":'/docs.json'
            }
        ],
        "static_url_path":"/flasgger_static",
        "swagger_ui":True,
        "specs_route":"/docs/"
    }

swagger = Swagger(app, template=swagger_template, config=swagger_config)

############################################### Database SQLITE ###################################################### 
con=sqlite3.connect("dbpredict_text.db")
con.execute("CREATE TABLE IF NOT EXISTS result(id INTEGER PRIMARY KEY AUTOINCREMENT ,text_ori TEXT NOT NULL, text_new TEXT NOT NULL, prediksi TEXT NOT NULL)")
con.close()

############################################### Open Model ###########################################################
tfidfVectorizer = joblib.load('tfidfVectorizer.pkl')
model_NN = joblib.load('classifier.pkl')

Tokenizer = joblib.load('Tokenizer.pkl') 
ModelLSTM = load_model('ModelLSTM.h5')
############################################### CRUD Based Website ################################################### 
@app.route('/')
def home():
    con=sqlite3.connect("dbpredict_text.db")
    con.row_factory=sqlite3.Row
    cur=con.cursor()
    cur.execute("SELECT * FROM result")
    data=cur.fetchall()
    con.close()
    return render_template("index.html",data=data) 

@app.route("/addDataNN",methods=["POST","GET"])
def addDataNN():
    if request.method=='POST':
        try:
            text_ori=request.form['text_ori'] 
            text_new=preprocessing(text_ori) 
            corpus = []
            corpus.append(text_new)
            text_features = tfidfVectorizer.transform(corpus).toarray()
            probability = model_NN.predict(text_features)
            tags = ['Negative','Netral','Positive']
            prediksi = tags[probability[0]] 
            con=sqlite3.connect("dbpredict_text.db")
            cur=con.cursor()
            cur.execute("INSERT INTO result(text_ori, text_new, prediksi)values(?,?,?)",(text_ori,text_new,prediksi))
            con.commit()
            flash("Record Added Successfully","success")
        except:
            
            flash("Error in Insert Operation","danger")
        finally:
            return redirect(url_for("home"))
            con.close() 

@app.route("/addDataLSTM",methods=["POST","GET"])
def addDataLSTM():
    if request.method=='POST':
        try:
            text_ori =  request.form.get('text_ori')
            text_new =preprocessing(text_ori)
            seq = Tokenizer.texts_to_sequences([text_new])
            padded = pad_sequences(seq, maxlen=128)
            probability = ModelLSTM.predict(padded)
            labels = ['Negative','Netral','Positive']
            prediksi = labels[np.argmax(probability)]
            con=sqlite3.connect("dbpredict_text.db")
            cur=con.cursor()
            cur.execute("INSERT INTO result(text_ori, text_new, prediksi)values(?,?,?)",(text_ori,text_new,prediksi))
            con.commit()
            flash("Record Added Successfully","success")
        except:
            
            flash("Error in Insert Operation","danger")
        finally:
            return redirect(url_for("home"))
            con.close()            

@app.route('/delete_record/<string:id>', methods=['GET'])
def delete_record(id):
    try:
        con = sqlite3.connect("dbpredict_text.db")
        cur = con.cursor()
        cur.execute("DELETE FROM result where id=?",(id))
        con.commit()
        flash("Record Deleted Successfully","success")
    except:
        flash("Record Delete Failed","danger")
    finally:
        return redirect(url_for("home"))
        con.close()
 
@app.route("/addFileNN", methods=['POST'])
def uploadFilesNN():
      # get the uploaded file
      uploaded_file = request.files['file']
      if uploaded_file.filename != '' and allowed_file(uploaded_file.filename):
           filename = secure_filename(uploaded_file.filename)
           file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
          # set the file path
           uploaded_file.save(file_path)
           parsedCSV(file_path)
          # save the file
      return redirect(url_for("home"))    
  
def parsedCSV(filePath):
      # CVS Column Names 
      # Use Pandas to parse the CSV file
      csvData = pd.read_csv(filePath)
      review = []
      for index, row in csvData.iterrows():
          review.append(preprocessing(row['text']))
          
      csvData['text cleaning'] = review  
      X_otherData = csvData['text cleaning']
      X_otherData = tfidfVectorizer.transform(X_otherData) 
      y_pred_otherData = model_NN.predict(X_otherData)
      csvData['result prediction'] = y_pred_otherData
      
      polarity_decode = {0 : 'Negative', 1 : 'Netral', 2 : 'Positive'}
      csvData['result prediction'] = csvData['result prediction'].map(polarity_decode)
      csvData.head()
      # Loop through the Rows
      for i,row in csvData.iterrows():
             con=sqlite3.connect("dbpredict_text.db")
             sql = "INSERT INTO result(text_ori, text_new, prediksi) VALUES (?, ?, ?)"
             value = (row['text'],row['text cleaning'],row['result prediction'])
             cur=con.cursor()
             cur.execute(sql, value)
             con.commit()
             print(i,row['text'],row['text cleaning'],row['result prediction'])

@app.route("/addFileNN", methods=['POST'])
def uploadFilesLSTM():
      # get the uploaded file
      uploaded_file = request.files['file']
      if uploaded_file.filename != '' and allowed_file(uploaded_file.filename):
           filename = secure_filename(uploaded_file.filename)
           file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
          # set the file path
           uploaded_file.save(file_path)
           parsCSV(file_path)
          # save the file
      return redirect(url_for("home"))    
  
def parsCSV(filePath):
      # CVS Column Names 
      # Use Pandas to parse the CSV file
      csvData = pd.read_csv(filePath)
      review = []
      for index, row in csvData.iterrows():
          review.append(preprocessing(row['text']))
          
      csvData['text cleaning'] = review  
      csvData['predicted'] = csvData['text cleaning'].apply(lambda x:find_sentiment([x]))
      csvData.head() 
      # Loop through the Rows
      for i,row in csvData.iterrows():
             con=sqlite3.connect("dbpredict_text.db")
             sql = "INSERT INTO result(text_ori, text_new, prediksi) VALUES (?, ?, ?)"
             value = (row['text'],row['text cleaning'],row['predicted'])
             cur=con.cursor()
             cur.execute(sql, value)
             con.commit()
             print(i,row['text'],row['text cleaning'],row['predicted'])
                            
############################################### Flask API Swagger Model NN ###################################################   

@swag_from("docs/ModelNN.yml", methods=['POST'])
@app.route("/Text_Neural_Network", methods=["POST"])
def Text_Neural_Network():
    text_ori =  request.form.get('text_ori')
    text_new =preprocessing(text_ori)
    corpus = []
    corpus.append(text_new)
    text_features = tfidfVectorizer.transform(corpus).toarray()
    probability = model_NN.predict(text_features)
    tags = ['Negative','Netral','Positive']
    prediksi = tags[probability[0]] 
    db = sqlite3.connect("dbpredict_text.db")
    cursor = db.cursor()
    sql = "INSERT INTO result(text_ori, text_new, prediksi) VALUES (?, ?, ?)"
    cursor = cursor.execute(sql, (text_ori, text_new, prediksi))
    db.commit()
    Text_Neural_Network = { 
			"text_original": text_ori,
			"text_new" : text_new,
            "prediksi" : prediksi 
		}
    return jsonify(Text_Neural_Network)

@swag_from("docs/ModelNN_File.yml", methods=['POST'])
@app.route("/fileNN", methods=['POST'])
def uploadFilesFromSwagger():
      # get the uploaded file
      uploaded_file = request.files['file']
      if uploaded_file.filename != '' and allowed_file(uploaded_file.filename):
           filename = secure_filename(uploaded_file.filename)
           file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
          # set the file path
           uploaded_file.save(file_path)
           parseCSV(file_path)
          # save the file
           db = sqlite3.connect("dbpredict_text.db")
           cursor = db.cursor()
           query = "SELECT id, text_ori, text_new, prediksi FROM result ORDER BY id DESC LIMIT 3"
           cursor.execute(query)
           texttweet = [
               dict(id = row[0], text_ori = row[1], text_new = row[2], prediksi = row[3])
               for row in cursor.fetchall()
              ]
      return jsonify(texttweet) 

def parseCSV(filePath):
      # CVS Column Names 
      # Use Pandas to parse the CSV file
      csvData = pd.read_csv(filePath)
      review = []
      for index, row in csvData.iterrows():
          review.append(preprocessing(row['text']))
          
      csvData['text cleaning'] = review  
      X_otherData = csvData['text cleaning']
      X_otherData = tfidfVectorizer.transform(X_otherData) 
      y_pred_otherData = model_NN.predict(X_otherData)
      csvData['result prediction'] = y_pred_otherData
      
      polarity_decode = {0 : 'Negative', 1 : 'Netral', 2 : 'Positive'}
      csvData['result prediction'] = csvData['result prediction'].map(polarity_decode)
      csvData.head()
      # Loop through the Rows
      for i,row in csvData.iterrows():
             con=sqlite3.connect("dbpredict_text.db")
             sql = "INSERT INTO result(text_ori, text_new, prediksi) VALUES (?, ?, ?)"
             value = (row['text'],row['text cleaning'],row['result prediction'])
             cur=con.cursor()
             cur.execute(sql, value)
             con.commit()
             print(i,row['text'],row['text cleaning'],row['result prediction'])
             
############################################### Flask API Swagger Model LSTM ################################################### 
@swag_from("docs/ModelLSTM.yml", methods=['POST'])
@app.route("/Text_LSTM", methods=["POST"])
def Text_LSTM():
    text_ori =  request.form.get('text_ori')
    text_new =preprocessing(text_ori)
    seq = Tokenizer.texts_to_sequences([text_new])
    padded = pad_sequences(seq, maxlen=128)
    probability = ModelLSTM.predict(padded)
    labels = ['Negative','Netral','Positive']
    prediksi = labels[np.argmax(probability)]
    db = sqlite3.connect("dbpredict_text.db")
    cursor = db.cursor()
    sql = "INSERT INTO result(text_ori, text_new, prediksi) VALUES (?, ?, ?)"
    cursor = cursor.execute(sql, (text_ori, text_new, prediksi))
    db.commit()
    Text_LSTM = { 
			"text_original": text_ori,
			"text_new" : text_new,
            "prediksi" : prediksi 
		}
    return jsonify(Text_LSTM)

@swag_from("docs/ModelLSTM_File.yml", methods=['POST'])
@app.route("/fileLSTM", methods=['POST'])
def uploadFilesFromLSTM():
      # get the uploaded file
      uploaded_file = request.files['file']
      if uploaded_file.filename != '' and allowed_file(uploaded_file.filename):
           filename = secure_filename(uploaded_file.filename)
           file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
          # set the file path
           uploaded_file.save(file_path)
           parsingCSV(file_path)
          # save the file
           db = sqlite3.connect("dbpredict_text.db")
           cursor = db.cursor()
           query = "SELECT id, text_ori, text_new, prediksi FROM result ORDER BY id DESC LIMIT 3"
           cursor.execute(query)
           texttweet = [
               dict(id = row[0], text_ori = row[1], text_new = row[2], prediksi = row[3])
               for row in cursor.fetchall()
              ]
      return jsonify(texttweet) 

def parsingCSV(filePath):
      # CVS Column Names 
      # Use Pandas to parse the CSV file
      csvData = pd.read_csv(filePath)
      review = []
      for index, row in csvData.iterrows():
          review.append(preprocessing(row['text']))
          
      csvData['text cleaning'] = review  
      csvData['predicted'] = csvData['text cleaning'].apply(lambda x:find_sentiment([x]))
      csvData.head()
      # Loop through the Rows
      for i,row in csvData.iterrows():
             con=sqlite3.connect("dbpredict_text.db")
             sql = "INSERT INTO result(text_ori, text_new, prediksi) VALUES (?, ?, ?)"
             value = (row['text'],row['text cleaning'],row['predicted'])
             cur=con.cursor()
             cur.execute(sql, value)
             con.commit()
             print(i,row['text'],row['text cleaning'],row['predicted'])
                           
if __name__ == '__main__':
    app.run(debug=True)
