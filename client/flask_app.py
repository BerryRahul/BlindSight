from flask import Flask, request
import os

app = Flask(__name__)

@app.route('/button', methods=['GET'])
def button_pressed():
    print("Button press detected from ESP32!")
    print("Running functionality triggered by button press...")

    
    os.system('python3 yolo.py')
    print('running yolo')
    os.system('python3 send_to_tts_esp.py')
    print('running tts')

    return "Button press acknowledged", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)