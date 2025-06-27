from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
with open('model/fake_news_model.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']
    vectorized_text = vectorizer.transform([news_text])
    prediction = model.predict(vectorized_text)[0]
    
    return jsonify({'result': prediction})

    
if __name__ == "__main__":
    app.run(debug=True)
