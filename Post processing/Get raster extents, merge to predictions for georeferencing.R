
library (raster)
library (rgdal)

# Input: Set working dir for ortho/mosaics
setwd(file.path('D:', 'SACR', '2023_mosaics_v3'))

# Inputs: pred1 = table from detection inference depicting predictions: xmin, ymin, w, h
# output_csv = name of output table of georeferenced predictions 
pred1 <- read.table ("D:/SACR/FINAL_2023_results/yolo_survey_pred_2023_v3.csv", sep = ",", header=TRUE, fill=TRUE)

output_csv = "D:/SACR/yolo_pred_survey_2023_v3b.csv"

## List files in dir and get extents
gridz <- list.files (pattern="*tif$")  
gridz

extents1 <- data.frame(raster_name=NA, xmin_extent=NA, xmax_extent=NA, ymin_extent=NA, ymax_extent=NA)
names(extents1)

extents1[1:5]


for (f in gridz){
  r1 <- raster(f)
  e1 <- extent(r1)
  raster_name <- f
  xmin_extent <- e1[1,]
  xmax_extent <- e1[2,]
  ymin_extent <- e1[3,]
  ymax_extent <- e1[4,]
  vector1 <- c(raster_name, xmin_extent, xmax_extent, ymin_extent, ymax_extent)
  print(vector1)
  extents1 <- rbind(extents1, vector1)
}

extents1$unique_image_jpg <- extents1$raster_name
extents1$raster_name <- NULL
View(extents1)

## Merge with predictions
pred1$basename <- basename(pred1$unique_image_jpg)
pred1$unique_image_jpg <- pred1$basename
pred1$basename <- NULL

merge5 <- merge(extents1, pred1, by="unique_image_jpg") 
merge5[0:5,]

write.table(merge5, "D:/SACR/yolo_pred_survey_2023_v3c.csv", sep=",", row.names=FALSE)
