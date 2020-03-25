from commons import transform_image, get_model, get_predictions

model = get_model()

def predict(image_bytes):
	tensor = transform_image(image_bytes=image_bytes)
	bboxes, labels, scores = get_predictions(tensor, model)
	return bboxes, labels, scores
