from flask import Flask, request
from analysis_network import analyze_text
app = Flask(__name__, static_url_path='/static')

@app.route('/')
def hello_world():
	return app.send_static_file('index.html')

@app.route('/vibe_check/<user_text>')
def test_network(user_text):
	assert user_text == request.view_args['user_text']

	print("Sending {} to server".format(user_text.lower()))

	return analyze_text(user_text.lower())


