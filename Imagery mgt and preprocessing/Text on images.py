import pandas
import cv2 as cv
from PIL import Image, ImageDraw, ImageFont

annotations = "D:/SACR/8_FINAL_2025_results/2_sacr_per_image_2025_surveys_conf20_gr5_cranes.csv"
image_path = "D:/SACR/8_FINAL_2025_results/survey_images_2025_conf20_gr5_cranes/"
export_path = "D:/SACR/8_FINAL_2025_results/images_w_text/"

annotations = pandas.read_csv(annotations)
# annotations.columns = (['image_id', 'xmin', 'ymin', 'w', 'h', 'label_id', 'unique_image_jpg'])

for index, row in annotations.iterrows():  ## iterrows: Pandas iterate over rows
    source_path = image_path + row['unique_image_jpg']
    image = Image.open(source_path)

    draw = ImageDraw.Draw(image)
    text = row['unique_image_jpg']
    font = ImageFont.truetype("arial.ttf", size = 25)
    position = (64, 36)
    print(position)
    color = (0, 0, 254)
    print(position, text, color, font)

    draw.text(xy = (360, 360), text = text, color = 112, font = font)

    draw.text(xy=(200, 30), text=text, color=112, font=font)
    draw.text(xy=(200, 100), text=text, color=112, font=font)
    draw.text(xy=(200, 300), text=text, color=112, font=font)
    draw.text(xy=(200,600), text=text, color=112, font=font)
    draw.text(xy=(200, 500), text=text, color=112, font=font)

    draw.text(xy=(800, 30), text=text, color=112, font=font)
    draw.text(xy=(800, 100), text=text, color=112, font=font)
    draw.text(xy=(800, 300), text=text, color=112, font=font)
    draw.text(xy=(800, 600), text=text, color=112, font=font)
    draw.text(xy=(800, 500), text=text, color=112, font=font)





    new_name = export_path + row['unique_image_jpg']
    print("new name: ", new_name)
    image.save(new_name)

