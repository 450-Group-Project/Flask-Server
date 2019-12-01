from flask import Flask, request, render_template
from analysis_network import analyze_text
app = Flask(__name__, static_url_path='/static')



# Home page
@app.route('/')
def main_page():
	return app.send_static_file('index.html')


def render_success():
	success_str = '''
	<div class="container">
	    <div class="card mb-4 text-white bg-success">
	      <div class="card-header">
	        <h4 class="my-0 font-weight-normal text-center">Positive</h4>
	      </div>
	      <div class="card-body text-center">
	        <form>
	          <div class="form-group">
	            <p>The text you entered was mostly positive!</p>
	          </div>
	        </form>
	        <button type="button" class="btn btn-lg btn-block btn-success">Analyze something else</button>
	      </div>
	    </div>
	  </div>
	'''

	return success_str


# Helper function to prep text for analysis
def clean_text(dirty_text):

	clean = ""

	allowed_chars = "abcdefghijklmnopqrstuvwxyz "

	for char in dirty_text.lower():
		if char in allowed_chars:
			clean += char

	if len(clean) >= 200:
		clean = clean[:199]

	return clean


@app.route('/vibe_check/<user_text>')
def test_network(user_text):

	assert user_text == request.view_args['user_text']

	cleaned_text = clean_text(user_text)

	print("Sending {} to server".format(user_text.lower()))

	score = float(analyze_text(user_text.lower()))

	# Negative
	if score <= .45:
		msg = "The text you entered was considered negative because it had a score less than 0.45"
		return render_template('negative_template.html', sentiment_score=score, message=msg, analyized_text=cleaned_text)

	# Neutral
	elif (score > 0.45) and (score < 0.55):
		msg = "The text you entered was considered neutral because it had a score between 0.45 and 0.55"
		return render_template('neutral_template.html', sentiment_score=score, message=msg, analyized_text=cleaned_text)

	# Positive
	else:
		msg = "The text you entered was considered positive because it had a score greater than 0.55"
		return render_template('positive_template.html', sentiment_score=score, message=msg, analyized_text=cleaned_text)

if __name__ == 'main':
    app.run(host="0.0.0.0", port=5000, debug=True)
