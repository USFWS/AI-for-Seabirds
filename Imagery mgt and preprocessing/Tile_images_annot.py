import os
import config
import detection_utils
from detection_utils import Slicer

# import image_bbox_tiler as ibis

#os.chdir("E:/a_detection_of_seabirds/new_tiles_test1")
## error in Main.py, line 273 is good;  line 300-305 is issue

# Inputs:
# im_src = image source
# an_src = annotation source
im_src = config.SOURCE_IMG
an_src = config.ANNOT_SOURCE

# Enter destination
im_dst = config.EXPORT_DIR
an_dst = config.ANNOT_EXPORT_DIR

print("ok")
im_list = os.listdir(im_src)
an_list = [x.replace("xml", "jpg") for x in os.listdir(an_src)]
list(set(im_list) - set(an_list))
list(set(an_list) - set(im_list))

#os.mkdir(im_dst)
#os.mkdir(an_dst)

slicer = detection_utils.Slicer()
slicer.config_dirs(img_src=im_src, ann_src=an_src, img_dst=im_dst, ann_dst=an_dst)
slicer.keep_partial_labels = True
slicer.save_before_after_map = True

# Slice images and annotationsl change empty sample
slicer.slice_by_size(tile_size=(1024,1024), tile_overlap=0.0, empty_sample=0.0)

# Slice images only
#slicer.slice_images_by_size((tile_size==(1024,1024), tile_overlap == 0.0))


#slicer.slice_by_size(tile_size=(1024,1024), tile_overlap=0.0, empty_sample= 0.0)