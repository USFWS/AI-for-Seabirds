import glob,os

import config

# Input dir where file extensions need to be changed
os.chdir(config.SOURCE_IMG)

source_dir = config.SOURCE_IMG
export_dir = "D:/seabird_detection/tiles_img2/"

if not os.path.exists(export_dir):
    os.makedirs(export_dir)

for filename in glob.glob('*..jpg'):
    print(filename)
    pre, ext = os.path.splitext(filename)
    rename1 = pre  + 'jpg'
    print("Rename: ", rename1)
    os.rename(os.path.join(source_dir, filename),
              os.path.join(export_dir, rename1))  # enter the new filename extension

#for filename in glob.glob('*.jpg.jpg'):
# print(filename)
#    pre, ext = os.path.splitext(filename)
#   rename1 = pre #+ '.jpg'
#  print ("Rename: ", rename1)
# os.rename(os.path.join(source_dir, filename),
#          os.path.join(export_dir, rename1)) # enter the new filename extension
#