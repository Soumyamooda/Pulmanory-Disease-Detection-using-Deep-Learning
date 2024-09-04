from flask import Flask, render_template, request, redirect, url_for, session
import os
import requests
from werkzeug.utils import secure_filename
import librosa as lb
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rdc_model
import re
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

# Load environment variables
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def format_phone_number(phone_number):
    phone_number = re.sub(r'\D', '', phone_number)  # Remove any non-digit characters
    if not phone_number.startswith('91'):
        phone_number = '91' + phone_number
    if not phone_number.startswith('+'):
        phone_number = '+' + phone_number
    return phone_number

def fetch_youtube_videos(query):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        'part': 'snippet',
        'q': query,
        'key': YOUTUBE_API_KEY,
        'maxResults': 5,
        'type': 'video'
    }
    response = requests.get(url, params=params)
    videos = []
    if response.status_code == 200:
        data = response.json()
        for item in data['items']:
            video = {
                'title': item['snippet']['title'],
                'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                'videoId': item['id']['videoId']
            }
            videos.append(video)
    return videos

root_folder = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER_temp = os.path.join(root_folder, "static")
UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER_temp, "uploads")
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key'

@app.route("/")
def dashboard():
    if 'user' in session:
        return render_template('dashboard.html')
    else:
        return redirect(url_for('login'))

@app.route("/predict")
def index():
    if 'user' in session:
        dir = UPLOAD_FOLDER
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
        return render_template("index.html", ospf=1)
    else:
        return redirect(url_for('login'))

@app.route("/", methods=['POST'])
def patient():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == "POST":
        plt.figure().clear()
        name = request.form["name"]
        lungSounds = request.files["lungSounds"]
        filename = secure_filename(lungSounds.filename)
        lungSounds.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        url2 = os.path.join("static", "uploads")
        url = os.path.join(url2, filename)
        absolute_url = os.path.abspath(url)
        
        res_list = rdc_model.classificationResults(absolute_url)
        if "Error" in res_list[0] or "No respiratory disorder detected" in res_list[0]:
            return render_template("index.html", ospf=0, n=name, lungSounds=url, res=res_list)
        
        try:
            audio1, sample_rate1 = lb.load(url, mono=True)
            librosa.display.waveshow(audio1, sr=sample_rate1, max_points=50000, x_axis='time', offset=0)
            plt.savefig("./static/uploads/outSoundWave.png")

            mfccs = lb.feature.mfcc(y=audio1, sr=sample_rate1, n_mfcc=40)
            fig, ax = plt.subplots()
            img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
            fig.colorbar(img, ax=ax)
            plt.savefig("./static/uploads/outSoundMFCC.png")

            url3 = os.path.join(url2, "outSoundWave.png")
            res_list.append(os.path.abspath(url3))
        except Exception as e:
            print(f"Error processing audio for visualisation: {e}")
            res_list.append("Error processing audio for visualization")

    return render_template("index.html", ospf=0, n=name, lungSounds=url, res=res_list)

@app.route("/precautions/<disease>")
def precautions(disease):
    if 'user' not in session:
        return redirect(url_for('login'))

    youtube_videos = fetch_youtube_videos(disease + ' precautions')
    return render_template("precautions.html", disease=disease, videos=youtube_videos)

@app.route("/send_sms", methods=['POST'])
def send_sms():
    if 'user' not in session:
        return redirect(url_for('login'))

    name = request.form.get('name')
    phone_number = request.form.get('phone_number')
    disease = request.form.get('disease')
    formatted_phone_number = format_phone_number(phone_number)
    message_body = f"In the report, the detected disease is {disease}"

    try:
        client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=formatted_phone_number
        )
        return f'''
            <script>
                alert("Message sent successfully to {name}!");
                window.location.href = "/";
            </script>
        '''
    except Exception as e:
        return str(e)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        # For demonstration, using hardcoded email and password
        if email == 'admin@example.com' and password == 'password':
            session['user'] = email
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid email or password')
    return render_template('login.html')

@app.route("/logout")
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run()
