from flask import Flask, render_template, request
app = Flask(__name__)


from commons import transform_image
from inference import predict


@app.route('/', methods  = ['GET', 'POST'])
def hello_world():
	if request.method == 'GET':
		return render_template('index.html', value = "with food")
	if request.method == 'POST':
		if 'file' not in request.files:
			print("file not uploaded")
			return
		file = request.files['file']
		image = file.read()
		bboxes, labels, scores = predict(image_bytes = image)
		return render_template('result.html', labels = labels, scores = scores, boxes = bboxes)

if __name__ == '__main__':
	app.run(debug=True)
