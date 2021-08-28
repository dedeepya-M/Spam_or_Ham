from flask import Flask, render_template, request, url_for
import pickle


app = Flask(__name__, template_folder='template')
my_model = pickle.load(open("spam_model.pkl", "rb"))
trans = pickle.load(open("count.pkl", "rb"))
@app.route("/")
def home():
    return render_template("page.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['msg']
        data = [text]
        counts = trans.transform(data).toarray()
        prediction = my_model.predict(counts)
        strn = ''
        if prediction == 1:
            strn = 'SPAM!!'
        else:
            strn = 'NOT SPAM'    
    return render_template("page.html", prediction_text=strn)

if __name__ == "__main__":
    app.run(debug=True)    
    
