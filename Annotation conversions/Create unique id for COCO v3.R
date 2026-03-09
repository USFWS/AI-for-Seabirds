
## When converting from csv to COCO annotation format, both a unique image 
## number and a unique annotation number are needed; this scripts creates
# those fields

library(dplyr)

#setwd(file.path('D:', 'SACR_models','2023_2025_SACR_model'))

setwd(file.path('D:', 'WHCR_2025', '12_WHCR_detection'))

data1 <- read.table ("2_whcr_annot_convert.csv", sep =",", header=TRUE, fill=TRUE)

#data1$id <- 0
data1$image_id <- data1$unique_image_jpg
# unique id per parent image
data1$image_id <- as.numeric(factor(data1$image_id))

data1$image_id  
# unique id per annotation
#data1$id <- data1 %>% group_indices (unique_BB)

View(data1)

write.table(data1,"2_whcr_annot_convert2.csv.csv", col.names=TRUE, row.names=FALSE, sep=",")
