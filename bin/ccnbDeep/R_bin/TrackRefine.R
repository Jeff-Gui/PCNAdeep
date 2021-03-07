# The script does two things: find potential parent-daughter cells, and wipes out
# random false classification in the form of A-B-A
# To determine potential parent-daughter cells, appearance and disappearance time
# and location of the track are examined. Tracks appearing within certain 
# distance and time shift after another track's disappearance is considered as the daughter track.
# Parent-daughter track does not necessary mean mitosis event. In fact, it can
# be caused by either of the three events
# - 1. cells moving outside the view field and then come back, therefore assigned differently
# - 2. A mis-transition: Ilastik assign cells belong to the same lineage to two tracks
# - 3. Temporal loss of signal or segmentation issue
# - 4. Mitosis

trackRefine = function(track, distance_tolerance, dist_factor, frame_tolerance, smooth){
  DIST_TOLERANCE = distance_tolerance # 50 Distance to search for parent-daughter relatoinship
  div_trans_factor = dist_factor # 2 recommanded
  FRAME_TOLERANCE = frame_tolerance # 15 Time distance to search for parent-daughter relationship
  window_length = smooth # 5 recommanded
  track = subset(track, track$trackId!=-1)
  
  #====================PART A: Transition refinement==================================
  #   Aim: correct false assignment of track A ID to adjacent track B when track A signal lost.
  #   Algorithm: detect false assignment by over-threshold frame-to-frame distance.
  # TODO: evaluate performance
  track_count = max(track$trackId)
  # Initialize current maximum track label. 
  # Any new track will be assigned with labels counted from the label.
  current_label = track_count
  track = track[order(track$trackId),] # sort by track ID
  for (i in 2:nrow(track)){
    if (dist(rbind(c(track$Center_of_the_object_0[i], track$Center_of_the_object_1[i]),
                   c(track$Center_of_the_object_0[i-1], track$Center_of_the_object_1[i-1]))) > DIST_TOLERANCE * (track$frame[i] - track$frame[i-1]) |
        (track$frame[i] - track$frame[i-1]) > FRAME_TOLERANCE){
      # when distance of object in track A at t=i to t=i-1 is larger than the threshold.
      if (track$trackId[i]==track$trackId[i-1]){
        # assign new track ID to track from t=[i] toward the end
        track$trackId[i:nrow(track)] = sub(paste("^", track$trackId[i-1],"$", sep = ""), 
                                           current_label+1, track$trackId[i:nrow(track)])
        track$lineageId[i:nrow(track)] = sub(paste("^", track$lineageId[i-1],"$", sep = ""), 
                                             current_label+1, track$lineageId[i:nrow(track)])
        current_label = current_label + 1 # update maximum track ID
      }
    }
  }
  track$trackId = as.numeric(track$trackId)
  track = track[order(track$trackId),]
  print(paste("Reorganized mis-transition objects:", current_label - track_count + 1))
  
  #=================PART B: Relationship prediction=========================
  # annotation table: record appearance and disappearance information of the track
  track_count = length(unique(track$trackId))
  ann = data.frame(
    "track" = unique(track$trackId),
    "app_frame" = rep.int(0, track_count), # appearance time frame
    "disapp_frame" = rep.int(0, track_count), # disappearance time frame
    "app_x" = rep.int(0, track_count), # appearance coordinate
    "app_y" = rep.int(0, track_count),
    "disapp_x" = rep.int(0, track_count), # disappearance coordinate
    "disapp_y" = rep.int(0, track_count),
    "app_stage" = rep_len(NA, track_count), # cell cycle classification at appearance
    "disapp_stage" = rep_len(NA, track_count), # cell cycle classification at disappearance
    "predicted_parent" = rep_len(NA, track_count), # non-mitotic parent track TO-predict
    "predicted_daughter" = rep_len(NA, track_count),
    "mitosis_parent" = rep_len(NA, track_count), # mitotic parent track to predict
    "mitosis_daughter" = rep_len(NA, track_count),
    "mitosis_identity" = rep_len(F, track_count)
    # TODO if a track is searched for mitosis and assigned to some daughters, it will not be searched for non-mitotic daughters
  )
  
  broken_tracks = vector()
  ids = unique(track$trackId)
  pars = unique(track$parentTrackId)
  ct2 = 0
  for (i in 1:length(ids)){
    cur_track = subset(track, track$trackId==ids[i])
    if (nrow(cur_track) >= FRAME_TOLERANCE | i %in% pars | cur_track$parentTrackId[1]!=0){
      # constraint: track < frame length tolerance is filtered out, No relationship can be deduced from that.
      ann$track[i] = ids[i]
      # (dis-)appearance time
      ann$app_frame[i] = min(cur_track$frame)
      ann$disapp_frame[i] = max(cur_track$frame)
      # (dis-)appearance coordinate
      ann$app_x[i] = cur_track$Center_of_the_object_0[1]
      ann$app_y[i] = cur_track$Center_of_the_object_1[1]
      ann$disapp_x[i] = cur_track$Center_of_the_object_0[nrow(cur_track)]
      ann$disapp_y[i] = cur_track$Center_of_the_object_1[nrow(cur_track)]
      # record (dis-)appearance cell cycle classification, in time range equals to FRAME_TOLERANCE
      if (nrow(cur_track)<FRAME_TOLERANCE){
        ann$app_stage[i] = paste(cur_track$predicted_class, collapse = ",")
        ann$disapp_stage[i] = paste(cur_track$predicted_class, collapse = ",")
      } else {
        ann$app_stage[i] = paste(cur_track$predicted_class[1:FRAME_TOLERANCE], collapse = ",")
        ann$disapp_stage[i] = paste(cur_track$predicted_class[(nrow(cur_track)-FRAME_TOLERANCE+1): nrow(cur_track)], collapse = ",")
      }
      
      # register mitosis prediction from that already had
      if (cur_track$parentTrackId[1]!=0){
        ct2 = ct2 + 1
        ann$mitosis_identity[i] = 'daughter'
        ann$mitosis_parent[i] = cur_track$parentTrackId[1]
        check = ann[which(ann$track==cur_track$parentTrackId[1]),'mitosis_daughter']
        if (is.na(check)){
          ann[which(ann$track==cur_track$parentTrackId[1]),'mitosis_daughter'] = ids[i]
        } else {
          ann[which(ann$track==cur_track$parentTrackId[1]),'mitosis_daughter'] = paste(as.character(check), as.character(ids[i]),sep='/')
        }
        ann[which(ann$track==cur_track$parentTrackId[1]),'mitosis_identity'] = 'parent'
      }
      
    } else {
      broken_tracks = c(broken_tracks, ids[i])
    }
  }
  ann = subset(ann, !ann$track%in%broken_tracks)
  track = subset(track, track$trackId%in%ann$track)
  print(paste("Daughters assigned during previous tracking:", ct2))
  print(paste("High quality tracks subjected to predict relationship:", nrow(ann)))
  
  count = 0
  # Mitosis search 1
  #   Aim: to identify two appearing daughter tracks after one disappearing parent track
  #   Algorithm: find potential daughters, for each pair of them, 
  potential_daughter_pair_id = intersect(ann$track[grep('M', ann$app_stage)], ann$track[which(ann$mitosis_identity==F)]) # daughter track must appear as M during mitosis
  for (i in 1:(length(potential_daughter_pair_id)-1)){
    for (j in (i+1):length(potential_daughter_pair_id)){
      # iterate over all pairs of potential daughters
      target_info_1 = ann[which(ann$track==potential_daughter_pair_id[i]),]
      target_info_2 = ann[which(ann$track==potential_daughter_pair_id[j]),]
      if (nrow(target_info_1)==0 | nrow(target_info_2)==0){ next }
      if (dist(rbind(
        c(target_info_1$app_x, target_info_1$app_y),
        c(target_info_2$app_x, target_info_2$app_y)
      )) <= (DIST_TOLERANCE * div_trans_factor) & abs(target_info_1$app_frame - target_info_2$app_frame) < FRAME_TOLERANCE){
        # Constraint A: close distance
        # Constraint B: close appearing time
        
        # Find potential parent that disappear at M
        potential_parent = intersect(ann$track[grep('M', ann$disapp_stage)], ann$track[which(ann$mitosis_identity==F)])
        if (length(potential_parent)>0){
          ann[which(ann$track==potential_daughter_pair_id[i]), "mitosis_identity"] = "daughter"
          ann[which(ann$track==potential_daughter_pair_id[j]), "mitosis_identity"] = "daughter"
          for (k in 1:length(potential_parent)){
            # spatial condition
            parent_x = ann[which(ann$track==potential_parent[k]), "disapp_x"]
            parent_y = ann[which(ann$track==potential_parent[k]), "disapp_y"]
            parent_disapp_time = ann[which(ann$track==potential_parent[k]), "disapp_frame"]
            parent_id = ann[which(ann$track==potential_parent[k]), "track"]
            if (dist(rbind(
              c(target_info_1$app_x, target_info_1$app_y),
              c(parent_x, parent_y)
            )) <= DIST_TOLERANCE * div_trans_factor &
            dist(rbind(
              c(parent_x, parent_y),
              c(target_info_2$app_x, target_info_2$app_y)
            )) <= DIST_TOLERANCE * div_trans_factor){
              # Constraint A: parent close to both daughter tracks' appearance
              if (abs(target_info_1$app_frame - parent_disapp_time) < FRAME_TOLERANCE &
                  abs(target_info_2$app_frame - parent_disapp_time) < FRAME_TOLERANCE){
                # Constraint B: parent disappearance time close to daughter's appearance
                # update information in ann table
                ann[which(ann$track==target_info_1$track), "mitosis_parent"] = parent_id
                ann[which(ann$track==target_info_2$track), "mitosis_parent"] = parent_id
                ann[which(ann$track==parent_id), "mitosis_identity"] = "parent"
                ann[which(ann$track==parent_id), "mitosis_daughter"] = paste(target_info_1$track, target_info_2$track, sep="/")
                # update information in track table
                for (t in which(track$trackId==target_info_1$track | track$trackId==target_info_2$track)){
                  track$lineageId[t] = parent_id
                  track$parentTrackId[t] = parent_id
                }
                for (t in which(track$lineageId==target_info_1$track | track$lineageId==target_info_2$track)){
                  track$lineageId[t] = parent_id
                }
                count = count + 1
              }
            }
          }
        }
      }
    }
  }
  print(paste("Low confidence mitosis relations found:", count))
  track = track[order(track$lineageId),]
  
  count = 0
  # Mitosis search 2: 
  #   Aim: solve mitotic track (daughter) that appear near another mitotic track (parent).
  #   Algorithm: find the pool of tracks that appear as mitotic. For each, find nearby mitotic tracks.
  sub_ann = subset(ann, ann$mitosis_identity == FALSE)
  potential_daughter_trackId = sub_ann$track[grep('M', sub_ann$app_stage)] # potential daughter tracks must appear at M phase during mitosis
  if (length(potential_daughter_trackId>0)){
    for (i in 1:length(potential_daughter_trackId)){
      target_info = ann[which(ann$track==potential_daughter_trackId[i]),]
      if (target_info$mitosis_identity != FALSE){next}
      
      # extract all info in the frame when potential daughter appears
      searching = subset(track, track$frame>=target_info$app_frame-floor(FRAME_TOLERANCE/3) & 
                           track$frame<=target_info$app_frame+floor(FRAME_TOLERANCE/3))
      # search for M cells (potential parent)
      searching = subset(searching, searching$predicted_class=="M" & searching$trackId!=potential_daughter_trackId[i])
      searching_filtered = data.frame()
      for (p in unique(searching$trackId)){
        if(ann[which(ann$track==p),'mitosis_identity'] == 'parent'){next}
        dif = abs(searching[searching$trackId==p,'frame']-target_info$app_frame)
        mf = searching[searching$trackId==p,'frame'][which(dif==min(dif))][1]
        searching_filtered = rbind(searching_filtered, searching[searching$trackId==p & searching$frame==mf,])
      }
      searching = searching_filtered
      
      if (nrow(searching)==0){ next }
      for (j in 1:nrow(searching)){
        if (dist(rbind(
          c(target_info$app_x, target_info$app_y),
          c(searching$Center_of_the_object_0[j], searching$Center_of_the_object_1[j])
        )) <= DIST_TOLERANCE * div_trans_factor){
          # Constraint: close distance
          if (!is.na(target_info["mitosis_parent"])){ 
            # if the potential daughter already has mitosis parent, will override.
            print(paste("Track",potential_daughter_trackId[i],"Warning: muiltiple mitosis parents found, only keep the last one."))}
          
          ann[which(ann$track==potential_daughter_trackId[i]),"mitosis_parent"] = searching$trackId[j]
          # label parent and daughter tracks as mitotic searched
          ann[which(ann$track==potential_daughter_trackId[i]),"mitosis_identity"] = "daughter"
          #print(paste("Daughter: ", potential_daughter_trackId[i], sep = ""))
          ann[which(ann$track==searching$trackId[j]), "mitosis_identity"] = "parent"
          ann[which(ann$track==searching$trackId[j]), "mitosis_daughter"] = potential_daughter_trackId[i]
          #print(paste("Parent: ", searching$trackId[j], sep = ""))
          # update lineage and parent track information of the daughter track
          for (k in which(track$trackId==target_info$track)){
            track$lineageId[k] = searching$trackId[j]
            track$parentTrackId[k] = searching$trackId[j]
          }
          for (k in which(track$lineageId==target_info$track)){
            track$lineageId[k] = searching$trackId[j]
          }
          count = count + 1
        }
      }
    }
  }
  print(paste("High confidence mitosis relations found:", count))
  track = track[order(track$lineageId),]
  
  count = 0
  # Lineage search
  #   Aim: to correct tracks with exactly one parent/daughter (gap-filling)
  #   Algorithm: first record parents and daughters for each single track, then link them up by the relationship.
  #   Key function: lineage search, works on tracks with exactly one parent/daughter
  lineage_search = function(tb,track_id,ln=vector()){
    # The function walks from the beginning of certain lineage till the end based 
    # works for one parent-daughter relationship
    # Input: tb(relationship annotation table), track_id: track ID to search
    # Output: list
    parents = tb$predicted_parent[which(tb$track==track_id)]
    daughter = tb$predicted_daughter[which(tb$track==track_id)]
    if (is.na(parents[1])){parents=c()}
    if (is.na(daughter[1])){daughter=c()}
    if (length(parents)<=1){
      # if the track has no parents or one parent
      if (length(daughter)==1){
        # if the track is the begin or in the middle of a lineage
        # daughter should not be mitosis
        if (length(grep("/", daughter))==0){
          ln = lineage_search(tb,daughter[1],c(ln,daughter[1]))
        }
      }
    }
    return(ln)
  }
  for (i in 1:nrow(ann)){
    # vectors to store predicted parents & daughters
    parent = vector()
    daughters = vector()
    # info of each iterated track
    cur_info = ann[i,]
    app_frame = cur_info$app_frame
    disapp_frame = cur_info$disapp_frame
    app_crd = c(cur_info$app_x, cur_info$app_y)
    disapp_crd = c(cur_info$disapp_x, cur_info$disapp_y)
    
    parent_range = (app_frame-FRAME_TOLERANCE):(app_frame-1)
    daughter_range = (disapp_frame+1):(disapp_frame+FRAME_TOLERANCE)
    # candidate parents and daughters drawn within frame tolerance range
    cdd_parent = subset(ann, ann$disapp_frame%in%parent_range)
    cdd_daughter = subset(ann, ann$app_frame%in%daughter_range)
    # verify parent relationship
    if (nrow(cdd_parent)>0 & cur_info$mitosis_identity!="daughter"){
      for (j in 1:nrow(cdd_parent)){
        cdd_crd = c(cdd_parent$disapp_x[j], cdd_parent$disapp_y[j])
        if (dist(rbind(app_crd,cdd_crd))<=DIST_TOLERANCE){
          # location constraint
          parent = c(parent, cdd_parent$track[j])
        }
      }
      if (length(parent)!=0){
        # store parent in the format "ID_a/ID_b/..."
        ann$predicted_parent[i] = paste(parent, collapse = "/")
        count = count + 1
      }
    }
    # verify daughter relationship
    if (nrow(cdd_daughter>0 & cur_info$mitosis_identity!="parent")){
      for (j in 1:nrow(cdd_daughter)){
        cdd_crd = c(cdd_daughter$app_x[j], cdd_daughter$app_y[j])
        if (dist(rbind(disapp_crd,cdd_crd))<=DIST_TOLERANCE){
          daughters = c(daughters, cdd_daughter$track[j])
        }
      }
      if(length(daughters)!=0){
        ann$predicted_daughter[i] = paste(daughters, collapse = "/")
        count = count + 1
      }
    }
  }
  print(paste("Lineage relations found: ", as.character(count), sep=""))
  # Based on predicted information, adjust track identity
  pool = c() # already searched track
  l = list()
  for (i in 1:nrow(ann)){
    if ((!is.na(ann$predicted_parent[i]) | !is.na(ann$predicted_daughter[i])) & is.na(ann$mitosis_parent[i])){
      if (!ann$track[i] %in% pool 
          & length(grep("/", ann$predicted_parent[i]))==0 
          | length(grep("/", ann$predicted_daughter[i]))==0){ 
        # only uni-parent and uni-daughter allowed to search, others should be mitosis track
        rlt = as.numeric(lineage_search(ann,ann$track[i],c(ann$track[i])))
        if (length(rlt)>=2){ l = c(l, list(rlt)); pool = c(pool, rlt) }
      }
    }
  }
  # for established lineage, assign parent track id to daughters
  for (lineage in l){
    lineage_v = lineage
    for (i in 2:length(lineage_v)){
      track[which(track$trackId==lineage_v[i]),"trackId"]=lineage_v[1]
      track[which(track$lineageId==lineage_v[i]),"lineageId"]=lineage_v[1]
      track[which(track$parentTrackId==lineage_v[i]), "parentTrackId"]=lineage_v[1]
    }
  }
  track = track[order(track$lineageId, track$trackId, track$frame),]
  print(paste("Lineage amount after reorganizing the lineage:", length(unique(track$lineageId))))
  
  #=================PART C: Classification Smoothing=========================
  track_filtered = track[c(),]
  padding = floor(window_length/2)
  for (i in unique(track$trackId)){
    cur_track = subset(track, track$trackId==i)
    if (nrow(cur_track)<window_length){
      track_filtered = rbind(track_filtered, cur_track)
      next
    }
    row_pad_begin = cur_track[1,]
    row_pad_end = cur_track[nrow(cur_track),]
    for (r in 1:padding){
      cur_track = rbind(row_pad_begin, cur_track)
      cur_track = rbind(cur_track, row_pad_end)
    }
    
    for (j in c("Probability.of.S", "Probability.of.M", "Probability.of.G1.G2")){
      cur_track[j] = stats::filter(cur_track[j], rep(1/window_length, window_length), sides = 2, method = "convolution")
    }
    escape = padding # escape the first some to make mitosis ends sharper
    range_pad = (padding+escape+1):(nrow(cur_track)-padding-escape)
    cur_track$predicted_class[range_pad] = sapply(range_pad, function(x){
      max_id = which(cur_track[x, c("Probability.of.S", "Probability.of.M", "Probability.of.G1.G2")] == 
                       max(cur_track[x, c("Probability.of.S", "Probability.of.M", "Probability.of.G1.G2")]))
      if (length(max_id)>1){
        # if tie, choose original one
        return(cur_track$predicted_class[x])
      }
      if (max_id==1){return("S")} else {
        if (max_id==2){return("M")} else {
          return("G1/G2")
        }
      }
    })
    range_pad = (padding+1):(nrow(cur_track)-padding)
    track_filtered = rbind(track_filtered, cur_track[range_pad,])
  }
  print(paste("Classification corrected by smoothing:", length(which(track_filtered$predicted_class!=track$predicted_class))))
  track = track_filtered
  
  return(track)
}

eu_dist = function(x_0, y_0, x_1, y_1){
  return(sqrt((x_0-x_1)**2 + (y_0-y_1)**2))
}

break_mitosis = function(track, dist_tolerance=40){
  # TrackMate plugin in Fiji stores daughters cells using same trackID, the function
  # separates them into two tracks.
  # TODO: optimize assignment matrix, generate distance cost matrix to solve more than 2 tracks (rare).
  track_count = max(track$trackId)
  for (i in unique(track$trackId)){
    cur_track = subset(track, track$trackId==i)
    if (length(unique(cur_track$frame)) != nrow(cur_track)){
      # search for frames including two tracks
      div_frame = cur_track$frame[which(diff(cur_track$frame)==0)[1]]
      daug1 = data.frame()
      daug1 = rbind(cur_track[which(cur_track$frame==div_frame)[1],])
      daug2 = data.frame()
      daug2 = rbind(cur_track[which(cur_track$frame==div_frame)[2],])
      j = div_frame + 1
      while (j <= max(cur_track$frame)){
        # update frame by frame using least distance
        x_daug1 = daug1$Center_of_the_object_0[nrow(daug1)]
        y_daug1 = daug1$Center_of_the_object_1[nrow(daug1)]
        x_daug2 = daug2$Center_of_the_object_0[nrow(daug2)]
        y_daug2 = daug2$Center_of_the_object_1[nrow(daug2)]
        cur_frame = subset(cur_track, cur_track$frame==j)
        if (nrow(cur_frame)==0){
          j = j + 1
          next
        }
        if (nrow(cur_frame)==1){
          cur_x = cur_frame$Center_of_the_object_0[1]
          cur_y = cur_frame$Center_of_the_object_1[1]
          if (eu_dist(cur_x, cur_y, x_daug1, y_daug1)/(j-daug1$frame[nrow(daug1)]) < 
              eu_dist(cur_x, cur_y, x_daug2, y_daug2)/(j-daug2$frame[nrow(daug2)])){
            if (eu_dist(cur_x, cur_y, x_daug1, y_daug1)/(j-daug1$frame[nrow(daug1)])<dist_tolerance){
              daug1 = rbind(daug1, cur_frame)
            }
          } else {
            if (eu_dist(cur_x, cur_y, x_daug2, y_daug2)/(j-daug2$frame[nrow(daug2)])<dist_tolerance){
              daug2 = rbind(daug2, cur_frame)
            }
          }
        } else {
          cur_x1 = cur_frame$Center_of_the_object_0[1]
          cur_y1 = cur_frame$Center_of_the_object_1[1]
          cur_x2 = cur_frame$Center_of_the_object_0[2]
          cur_y2 = cur_frame$Center_of_the_object_1[2]
          if (eu_dist(cur_x1, cur_y1, x_daug1, y_daug1) < eu_dist(cur_x2, cur_y2, x_daug1, y_daug1)){
            # assign 1 to 1
            if (eu_dist(cur_x1, cur_y1, x_daug1, y_daug1)/(j-daug1$frame[nrow(daug1)])<dist_tolerance){
              daug1 = rbind(daug1, cur_frame[1,])
              daug2 = rbind(daug2, cur_frame[2,])
            }
          } else {
            if (eu_dist(cur_x2, cur_y2, x_daug1, y_daug1)/(j-daug1$frame[nrow(daug1)])<dist_tolerance){
              daug1 = rbind(daug1, cur_frame[2,])
              daug2 = rbind(daug2, cur_frame[1,])
            }
          }
        }
        j = j + 1
      }
      daug1$trackId = track_count + 1
      daug2$trackId = track_count + 2
      daug1$parentTrackId = i
      daug2$parentTrackId = i
      track_count = track_count + 2
      track = track[-which(track$trackId==i & track$frame>=div_frame),]
      track = rbind(track, daug1)
      track = rbind(track, daug2)
    }
  }
  return(track)
}


