# The script merges object classification and tracking outputs from ilastik.
# The script is adapted from previous one using for foci-based classification
# therefore has vague variable names. In general, "foci" refers to object
# classification output.
# The script screens for each frame and matches object identity to tracked object
# based on their spatial relation (object center coordinates).

mergeTrackAndClass = function(track,foci,elective=NULL){
  library(dplyr)
  track$POSITION_X = round(track$POSITION_X)
  track$POSITION_Y = round(track$POSITION_Y)
  foci$Center_of_the_object_0 = round(foci$Center_of_the_object_0)
  foci$Center_of_the_object_1 = round(foci$Center_of_the_object_1)
  test = inner_join(foci, track, by=c("frame"="FRAME",
                                      "Center_of_the_object_0"="POSITION_Y",
                                      "Center_of_the_object_1"="POSITION_X"))
  n = colnames(test)
  n[which(n=="phase")] = "predicted_class"
  n[which(n=="TRACK_ID")] = "trackId"
  n[which(n=="Probability.of.I")] = "Probability.of.G1.G2"
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
  
  # relabel track sequentially from 1
  ori_ids = unique(test$trackId)
  track_count = length(ori_ids)
  dic = as.vector(1:track_count)
  names(dic) = ori_ids[order(ori_ids)]
  test$trackId = dic[as.character(test$trackId)]
  test['lineageId'] = test$trackId
  test['parentTrackId'] = rep(0, nrow(test))  # TODO: check why LAP tracker do not output parent relationships
  test['Probability.of.S'] = rep(0, nrow(test))
  test = test[,c('frame','trackId','lineageId','parentTrackId','Center_of_the_object_0','Center_of_the_object_1','predicted_class','Probability.of.S','Probability.of.G1.G2','Probability.of.M', elective_cols)]
  test$predicted_class = gsub('I','G1/G2', test$predicted_class)
  return(test)
}

