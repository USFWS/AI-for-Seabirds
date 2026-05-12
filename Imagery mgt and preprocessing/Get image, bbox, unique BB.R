
library(dplyr)
library(stringr)
library(tidyverse)

setwd(file.path('C:', 'users', 'aware', 'desktop',
                'whcr_results'))

###### Get parent image names and coordinates
## be sure to remove rows that have non-jpg crop names
data1<- read.table("YOLO_to_format.csv", sep=",", header=TRUE)

names(data1)
View(data1)

data1$basename <- basename(data1$unique_image_jpg)
data1$basename = substr(data1$basename,1,nchar(data1$basename)-4)
data1$basename

data1 <- separate_wider_delim (data1, cols= "basename", delim="_", names= c("c1", "c2", "c3", "c4", "c5", "c6", "xmin", "ymin", "w", "h"), too_many="drop")

data1[0:5,]

data1$unique_image_jpg <- paste(data1$c1, data1$c2, data1$c3, data1$c4,
                                data1$c5, data1$c6, sep= "_")

data1$unique_image_jpg <- paste0(data1$unique_image_jpg) #, ".jpg")
data1$unique_image_jpg

data1$unique_BB <- paste0(data1$basename, "_", data1$xmin, "_", data1$ymin, "_", data1$w, "_", 
                          data1$h, ".jpg")
data1$unique_BB

data1$c1 <- NULL
data1$c2 <- NULL
data1$c3 <- NULL
data1$c4 <- NULL
data1$c5 <- NULL
data1$c6 <- NULL

View(data1)

write.table(data1, "new1.csv", sep =",", row.names=FALSE)
