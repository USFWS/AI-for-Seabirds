
library(dplyr)

# Set directory                         
setwd(file.path('C:', 'Users', 'bpickens','OneDrive - DOI', 'Seabird_workflow', 'Metadata_images_cumulative'))

# Input metadata to shorten
data1 <- read.table ("2025_image_metadata_update_Aug2025.csv",sep = ",",header=TRUE, fill=TRUE)

# name of output csv
output_csv <- "2025_metadata_short_update_Aug2025.csv"

# Methods to shorten
data1$flight_name <- NULL
data1$camera_GUID <- NULL
data1$roll <- NULL
data1$pitch <- NULL
data1$yaw <- NULL
data1$speed <- NULL
data1$rmse <- NULL
data1$velocityD <- NULL
data1$velocityE <- NULL
data1$velocityN <- NULL
data1$frame_number <- NULL
data1$lat <- NULL
data1$long <- NULL
data1$BLLat <- NULL
data1$BLLong <- NULL
data1$BRLat <- NULL
data1$BRLong <- NULL
data1$TRLat <- NULL
data1$TRLong <- NULL
data1$TLLat <- NULL
data1$TLLong <- NULL
data1$flight_line <- NULL
data1$unique_image_jpg <- paste(data1$unique_image,".jpg", sep = "")
data1$unique_image_jpg

write.table(data1, output_csv, sep=",", row.names=FALSE)

#################################
### Combine short metadata

data1 <- read.table ("2021_2024_image_metadata_corrected_short_v6.csv",sep = ",",header=TRUE, fill=TRUE)

data2 <- read.table ("2025_metadata_short_update_Aug2025.csv",sep = ",",header=TRUE, fill=TRUE)

data3 <- rbind(data1, data2)

write.table(data3, "2021_2025_image_metadata_short_v7.csv", sep=",", row.names=FALSE)


