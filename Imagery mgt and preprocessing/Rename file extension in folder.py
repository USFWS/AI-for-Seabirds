import glob,os

# Input dir where file extensions need to be changed
os.chdir("D:/Hive_batch4/Priority1_background_images/")

source_dir = "D:/Hive_batch4/Priority1_background_images/"
export_dir = "D:/Hive_batch4/Priority1_background_images/"


if not os.path.exists(export_dir):
    os.makedirs(export_dir)

for filename in glob.glob('*.JPG'):
    print(filename)
    pre, ext = os.path.splitext(filename)
    rename1 = pre  + '.jpg'
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