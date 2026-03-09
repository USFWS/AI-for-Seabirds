
## When converting from csv to COCO annotation format, both a unique image 
## number and a unique annotation number are needed; this scripts creates
# those fields

library(dplyr)

setwd(file.path('D:', 'SACR_models','2023_2025_SACR_model'))

data1 <- read.table ("status_annot_new.csv", 
            sep =",", header=TRUE, fill=TRUE)

data1$id <- 0
data1$image_id <- 0

names(data1)
View(data1)

# unique id per parent image
data1$image_id <- data1 %>% group_indices (unique_image_jpg)  
  
# unique id per annotation
data1$id <- data1 %>% group_indices (unique_BB)


write.table(data1,"2023_yolov5_new_annot2.csv", col.names=TRUE, row.names=FALSE, sep=",")
