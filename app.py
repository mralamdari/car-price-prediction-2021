import os, flask, pickle, numpy as np, pandas as pd
from sklearn import preprocessing


app = flask.Flask(__name__)
app.config["SECRET_KEY"] = "mykey"
img_folder = os.path.join('static', 'img')
app.config['UPLOAD_FOLDER'] = img_folder

@app.route("/")
# @app.route("/", methods=["POST"])
# @app.route("/index", methods=["POST"])
def index():
    red_car = os.path.join(app.config['UPLOAD_FOLDER'], 'red_car.jpg')
    return flask.render_template("index.html", red=red_car)


@app.route("/help")
def help():
    blue_car = os.path.join(app.config['UPLOAD_FOLDER'], 'blue_car.jpg')
    return flask.render_template("help.html", blue=blue_car)


@app.route("/about")
def about():
    silver_car = os.path.join(app.config['UPLOAD_FOLDER'], 'silver_car.jpg')
    return flask.render_template("about.html", silver=silver_car)

@app.route("/results", methods=["GET", "POST"])
def results():
    white_car = os.path.join(app.config['UPLOAD_FOLDER'], 'white_car.jpg')
    try:
        carwidth = float(flask.request.args.get("carwidth"))
        curbweight = float(flask.request.args.get("curbweight"))
        enginesize = float(flask.request.args.get("enginesize"))
        compressionratio = float(flask.request.args.get("compressionratio"))
        horsepower = float(flask.request.args.get("horsepower"))
        highwaympg = float(flask.request.args.get("highwaympg"))
        citympg = float(flask.request.args.get("citympg"))    
        fueleconomy=(highwaympg + citympg)//2

        input_list = np.array([carwidth, curbweight, enginesize, compressionratio,horsepower, highwaympg, fueleconomy, citympg])
        loadm = pickle.load(open("h2_all_best_model.sav", "rb"))
        le = preprocessing.MinMaxScaler()
        test = pd.DataFrame(le.fit_transform(input_list.reshape(-1, 1))).T
        pred=loadm.predict(test)
    except TypeError:
        return flask.render_template("index.html", white=white_car)

    return flask.render_template("results.html", white=white_car, res=round(pred[0], 4))

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=os.environ.get("PORT", 5000))   
