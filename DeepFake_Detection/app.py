from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from app.inference import prepare_single_video
from app.utils import allowed_file

UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = tf.keras.models.load_model('./model/deepfake_video_model.h5')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        
        frame_feat, frame_mask = prepare_single_video(path)
        prediction = model.predict([frame_feat, frame_mask])[0][0]
        
        result = "FAKE" if prediction >= 0.5 else "REAL"
        os.remove(path)
        return jsonify({'result': result, 'confidence': float(prediction)})
    else:
        return jsonify({'error': 'Invalid file type'}), 400
    

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)