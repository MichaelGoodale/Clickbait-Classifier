from flask import Flask, request, render_template, flash
from wtforms import Form, StringField, validators
from grpc_functions import grpc_get

SERVING_IP = "localhost:9000"
SENTENCE_MAX = 30
WORD_VECTOR_SIZE = 50

app = Flask(__name__)
app.secret_key = "super secret key"
    
@app.route("/", methods = ["GET", "POST"])
def home_page():
    form = ClickbaitForm(request.form)
    scores = [0,0]
    if request.args.get("headline") is not None:
        scores, classes = grpc_get(SERVING_IP, request.args.get("headline"), SENTENCE_MAX, WORD_VECTOR_SIZE)
        values = []
        for i, o in enumerate(classes):
            values.append(str(round(scores[i]*100, 2))+"% "+str(o).lower())
        flash("\""+request.args.get("headline") + "\" is "+values[0]+" and "+values[1])
    return render_template('index.html', form=form, percentClickbait = int(round(scores[1]*100, 2)))

@app.route("/headlines/<head>")
def show_headline(head):
    return "The headline was %s" % head

if __name__ == "__main__":
    app.config['SESSION_TYPE'] = 'filesystem'
    sess.init_app(app)

    app.run()
    
class ClickbaitForm(Form):
    headline = StringField("Enter a Headline!", [validators.Length(min=4, max=25)])
