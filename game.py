# To Capture Frame
import cv2

# To process image array
import numpy as np

# To Load the Pre-trained Model
import tensorflow as tf

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Loading the pre-trained model : keras_model.h5
mymodel = tf.keras.models.load_model('keras_model.h5')

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)

		# Resize the frame
		resized_frame = cv2.resize(frame , (224,224))

		# Expanding the dimension of the array along axis 0
		resized_frame = np.expand_dims(resized_frame , axis = 0)

		# Normalizing for easy processing
		resized_frame = resized_frame / 255

		# Getting predictions from the model
		predictions = mymodel.predict(resized_frame)

		# Converting the data in the array to percentage confidence 
		rock = int(predictions[0][0]*100)
		paper = int(predictions[0][1]*100)
		scissor = int(predictions[0][2]*100)

		# printing percentage confidence
		print(f"Rock: {rock} %, Paper: {paper} %, Scissor: {scissor} %")

		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()