import cv2
import numpy as np 
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--play_video', help="Tue/False", default=True)
parser.add_argument('--image', help="Tue/False", default=False)
parser.add_argument('--video_path', help="Path of video file", default="videos/wa.mp4")
parser.add_argument('--image_path', help="Path of image to detect objects", default="Images/bicycle.jpg")
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()

shrimp_timeline = []

#Load yolo
def load_yolo():
	net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
	classes = []
	with open("coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]

	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 400, size=(len(classes), 3))
	return net, classes, colors, output_layers

def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	return img, height, width, channels

def start_webcam():
	cap = cv2.VideoCapture(0)

	return cap


def display_blob(blob):
	'''
		Three images each for RED, GREEN, BLUE channel
	'''
	for b in blob:
		for n, imgb in enumerate(b):
			cv2.imshow(str(n), imgb)

def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)

	return boxes, confs, class_ids

def box_diff(box1, box2):
	x, y, w, h = box1
	x1, y1, w1, h1 = box2

	return (x1 - x) + (y1 - y) + (w1 - w) + (h1 - h)

def measure_class():
	THRESHOLD = 100
	frame_count = len(shrimp_timeline)
	if frame_count < 10:
		return 0 # Not moving shrimp

	accumulation_diff = 0
	for i in range(frame_count - 10, frame_count - 1):
		accumulation_diff += abs(box_diff(shrimp_timeline[i], shrimp_timeline[i + 1]))

	print(accumulation_diff)
	if (accumulation_diff < 0):
		accumulation_diff = -accumulation_diff

	return accumulation_diff

def draw_labels(boxes, confs, colors, class_ids, classes, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	eating=0
	countbox=0
	if confs:
		fish_count = 0
		acc = measure_class()
		for i in range(len(boxes)):
			
			if i in indexes:
				fish_count += 1
				x, y, w, h = boxes[i]
				shrimp_timeline.append(boxes[i])
				acc_value = measure_class()
				#acc = 
				label = "avg speed is " + str(int(acc_value / 100))
				cv2.rectangle(img, (0, 3), (200, 75), (0, 0, 0), 1)
				cv2.putText(img, "           ,"  + "avg speed " + str(int(acc/100)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 25, 25), 4)

				if y < 150:
					eating += 1
				if eating == 0:
					cv2.putText(img, "                                                    "+"Fish not eating ", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 77, 77), 4)
			

				color = colors[i]
				cv2.rectangle(img, (x,y), (x+w, y+h), color, 5)
				cv2.putText(img, label, (x, y - 5), font, 2, color, 2)
		print(colors[3])
		cv2.rectangle(img, (0, 0), (200, 75), (0, 0, 0), -1)
		cv2.putText(img, "Count is " +  str(fish_count), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 77, 77), 4)
		#cv2.putText(img, "avg speed " + str(acc_value/100), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (300, 300, 300), 2)
		cv2.putText(img, "                                     "+"Eating " +  str(eating), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 77, 77), 4)
		cv2.imshow("Image", img)

def image_detect(img_path): 
	model, classes, colors, output_layers = load_yolo()
	image, height, width, channels = load_image(img_path)
	blob, outputs = detect_objects(image, model, output_layers)
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
	draw_labels(boxes, confs, colors, class_ids, classes, image)
	
	while True:
		
		key = cv2.waitKey(1)
		if key == 27:
			break


def webcam_detect():
	model, classes, colors, output_layers = load_yolo()
	cap = start_webcam()
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()


def start_video(video_path):
	model, classes, colors, output_layers = load_yolo()
	cap = cv2.VideoCapture(video_path)

	shrimp_timeline = []
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
	
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()



if __name__ == '__main__':
	webcam = args.webcam
	video_play = args.play_video
	image = args.image
	if webcam:
		if args.verbose:
			print('---- Starting Web Cam object detection ----')
		webcam_detect()
	if video_play:
		video_path = args.video_path
		if args.verbose:
			print('Opening '+video_path+" .... ")
		start_video(video_path)
	if image:
		image_path = args.image_path
		if args.verbose:
			print("Opening "+image_path+" .... ")
		image_detect(image_path)
	

	cv2.destroyAllWindows()