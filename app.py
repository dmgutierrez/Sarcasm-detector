from flask import Flask, render_template, request
import os
from helper import global_variables as gv


app_title = os.getenv('APP_TITLE', 'Welcome to the Sarcasm detector!')
os.environ["GLOBAL_PARENT_DIR"] = ""
base_url = "http://" + str(gv.host) + ":" + str(gv.port)
url_sarcasm_sentiment = base_url + "/sarcasm/sentiment"
app = Flask(__name__,
            static_folder=os.path.join('www', 'static'),
            template_folder=os.path.join('www', "templates"))
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', app_title=app_title,
                           url_sarcasm_sentiment=url_sarcasm_sentiment)


@app.route('/sarcasm/sentiment', methods=['GET', 'POST'])
def sarcasm_sentiment_request():
    if request.method == 'POST':
        sentence_input = request.form["input"]

        # ==========================================================
        response = gv.sarcasm_service.launch_analyzer(sentence_input)
        # ==========================================================
        return render_template('results.html', app_title=app_title, confidence=response["confidence"],
                               sarcasm=response["value"], user_input=response["sentence"])
    else:
        render_template('index.html', app_title=app_title,
                        url_sarcasm_sentiment=url_sarcasm_sentiment)


if __name__ == '__main__':
    gv.init_logger_object()
    gv.init_service()
    app.run(host=gv.host, port=gv.port, passthrough_errors=False)