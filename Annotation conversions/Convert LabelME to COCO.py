import labelme2coco

# Inputs: annot_dir = dir that contains labelme annotation files;
#  export_dir = dir to save newly formatted annotations

annot_dir = "D:/seabird_detection/hive_batch4_priority4_species/"
export_dir = "D:/seabird_detection/hive_batch4_annot/"

# convert labelme annotations to coco annotations
labelme2coco.convert(annot_dir, export_dir)