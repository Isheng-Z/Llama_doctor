import os
import json
import base64
import time
import wave
import pyaudio
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import urlencode
from flask import Flask, jsonify
from flask_cors import CORS

# ================= Recording Parameters =================
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1  # Mono
RATE = 16000  # 16k sample rate (Required by Baidu API)
CHUNK = 1024  # Audio data chunk size
RECORD_SECONDS = 10  # Recording duration (seconds)
AUDIO_FILENAME = "audio.wav"  # Saved audio file name

# ================= Baidu API Parameters =================
API_KEY = "tTNDIpXO2hgOOcMiRIep9CSc"
SECRET_KEY = "tUMVhc0VpWOOwX5wNv8sJE6aLkbbgxy1"

ASR_URL = 'http://vop.baidu.com/server_api'
TOKEN_URL = 'http://aip.baidubce.com/oauth/2.0/token'
DEV_PID = 1737  # 1737: English input model
CUID = '123456PYTHON'

# Flask server
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests


class DemoError(Exception):
    pass


def record_audio():
    """Record audio and save as audio.wav"""
    print("🎤 Recording started. Please speak...")

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):  # Record for RECORD_SECONDS seconds
        data = stream.read(CHUNK)
        frames.append(data)

    print("🎤 Recording finished. Saving...")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save as WAV file
    with wave.open(AUDIO_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"✅ Audio saved to: {AUDIO_FILENAME}")


def fetch_token():
    """Get access token from Baidu API"""
    params = {'grant_type': 'client_credentials',
              'client_id': API_KEY,
              'client_secret': SECRET_KEY}
    post_data = urlencode(params).encode('utf-8')

    req = Request(TOKEN_URL, post_data)
    try:
        f = urlopen(req)
        result_str = f.read().decode()
    except URLError as err:
        print(f'❌ Failed to get token: {err.code}')
        result_str = err.read()

    result = json.loads(result_str)
    if 'access_token' in result and 'scope' in result:
        print(f'✅ Token acquired: {result["access_token"]}')
        return result['access_token']
    else:
        raise DemoError('❌ Failed to obtain token. Please check your API_KEY and SECRET_KEY')


def recognize_audio():
    """Use Baidu Speech Recognition API to recognize audio"""
    # Read audio data
    with open(AUDIO_FILENAME, 'rb') as speech_file:
        speech_data = speech_file.read()

    length = len(speech_data)
    if length == 0:
        raise DemoError(f'❌ Audio file {AUDIO_FILENAME} is empty. Please check if recording succeeded.')

    # Base64 encode
    speech = base64.b64encode(speech_data).decode('utf-8')

    token = fetch_token()
    params = {
        'dev_pid': DEV_PID,
        'format': "wav",
        'rate': RATE,
        'token': token,
        'cuid': CUID,
        'channel': 1,
        'speech': speech,
        'len': length
    }

    post_data = json.dumps(params, sort_keys=False).encode('utf-8')
    req = Request(ASR_URL, post_data)
    req.add_header('Content-Type', 'application/json')

    try:
        print("📡 Sending audio data to Baidu Speech Recognition API...")
        begin = time.perf_counter()
        f = urlopen(req)
        result_str = f.read().decode()
        print(f"✅ API request completed in {time.perf_counter() - begin:.2f} seconds")
    except URLError as err:
        print(f'❌ Speech recognition API request failed: {err.code}')
        result_str = err.read()

    # Parse result
    result = json.loads(result_str)
    if result["err_no"] == 0:
        recognized_text = result["result"][0]
        print(f"🎯 Recognition result: {recognized_text}")
        return recognized_text
    else:
        print(f"❌ Recognition failed. Error code: {result['err_no']}")
        return None


# ================= Flask API Endpoint =================
@app.route('/speech-to-text', methods=['GET'])
def speech_to_text():
    """Flask API endpoint: Record -> Recognize"""
    try:
        record_audio()  # Record audio
        recognized_text = recognize_audio()  # Recognize audio

        if recognized_text:
            return jsonify({"success": True, "text": recognized_text})
        else:
            return jsonify({"success": False, "message": "Speech recognition failed"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ================= Run Flask Server =================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
