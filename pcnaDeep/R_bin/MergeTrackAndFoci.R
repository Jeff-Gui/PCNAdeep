# The script merges object classification and tracking outputs from ilastik.
# The script is adapted from previous one using for foci-based classification
# therefore has vague variable names. In general, "foci" refers to object
# classification output.
# The script screens for each frame and matches object identity to tracked object
# based on their spatial relation (object center coordinates).

mergeTrackAndFoci = function(track,foci,elective=NULL){
  library(dplyr)
  track$Center_of_the_object_0 = floor(track$Center_of_the_object_0)
  track$Center_of_the_object_1 = floor(track$Center_of_the_object_1)
  test = inner_join(track, foci, by=c("frame"="frame",
                                      "Center_of_the_object_0"="Center_of_the_object_0",
                                      "Center_of_the_object_1"="Center_of_the_object_1"))
  n = colnames(test)
  n[which(n=="phase")] = "predicted_class"
  colnames(test) = n
  elective_cols = elective
  for(e in elective_cols){
    if(! e %in% n){
      print("Error! Elective field not in input table")
      exit()
    }
  }
  temp = test$Center_of_the_object_0
  test$Center_of_the_object_0 = test$Center_of_the_object_1
  test$Center_of_the_object_1 = temp
  return(test[,c('frame','trackId','lineageId','parentTrackId','Center_of_the_object_0','Center_of_the_object_1','predicted_class','Probability.of.S','Probability.of.G1.G2','Probability.of.M', elective_cols)])
}

