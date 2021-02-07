# The script merges object classification and tracking outputs from ilastik.
# The script is adapted from previous one using for foci-based classification
# therefore has vague variable names. In general, "foci" refers to object
# classification output.
# The script screens for each frame and matches object identity to tracked object
# based on their spatial relation (object center coordinates).

mergeTrackAndFoci = function(track,foci){
  foci$Center.of.the.object_0 = floor(foci$Center.of.the.object_0)
  foci$Center.of.the.object_1 = floor(foci$Center.of.the.object_1)
  track$Object_Center_0 = floor(track$Object_Center_0)
  track$Object_Center_1 = floor(track$Object_Center_1)
  library(dplyr)
  test = inner_join(track, foci, by=c("frame"="timestep",
                                      "Object_Center_0"="Center.of.the.object_0",
                                      "Object_Center_1"="Center.of.the.object_1"))
  test_coln = colnames(test)
  test_coln[which(test_coln=="Object_Center_0")] = "Center_of_the_object_0"
  test_coln[which(test_coln=="Object_Center_1")] = "Center_of_the_object_1"
  if (length(which(test_coln=="Predicted.Class"))>0){
    test_coln[which(test_coln=="Predicted.Class")] = "predicted_class"
  } else {
    if (length(which(test_coln=="predicted_class"))==0){
      print("Warning! Column name of predicted class is ambiguous.")
    }
  }
  colnames(test) = test_coln
  return(test[,c('frame','trackId','lineageId','parentTrackId','Center_of_the_object_0','Center_of_the_object_1','predicted_class','Probability.of.S','Probability.of.G1.G2','Probability.of.M')])
}

