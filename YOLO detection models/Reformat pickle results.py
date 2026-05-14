import pickle
import pandas as pd
import csv
import os

pickle_dir = "C:/BP/demo/demo_results/exp/pickles/"
new_csv = "C:/BP/demo/demo_results/new.csv"

# pickle_dir = "E:/WHCR_2025/detection/demo_results/exp5/pickles/"
#new_csv = "E:/WHCR_2025/detection/demo_results/exp5/new.csv"

pickle_dir = os.path.dirname(pickle_dir)

with open(new_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['unique_image_jpg', "xmin", "ymin", "w", "h", "class", "score"])
x=0
for root, dirs, files in os.walk(pickle_dir):
	for filename in files:

		#print(filename)
		path = os.path.join(root, filename)
		#print(path)
		list_file = open(path, 'rb')
		list2 = pickle.load(list_file)
		total = len(list2)
		#total = sum(isinstance(i, list2) for i in list2)
		print("total", total)
		basename = os.path.basename(path)
		basename = basename.removesuffix('.pickle')
		basename = basename + ".jpg"
		#print("basename: ", basename)

		for i in range(0, total):
			score = list2[i].score.value
			cat = list2[i].category.id
			bbox = list2[i].bbox.to_coco_bbox()
			xmin = bbox[0]
			xmin = int(xmin)
			ymin = bbox[1]
			ymin = int(ymin)
			w = bbox[2]
			w = int(w)
			h = bbox[3]
			h = int(h)
			print("cool", xmin, ymin, w, h)

			print(basename, bbox, cat, score)
			with open(new_csv, 'a', newline='') as file:
				writer = csv.writer(file)
				for j in range(1, total):
					writer.writerow([basename, xmin, ymin, w, h, cat, score])

#