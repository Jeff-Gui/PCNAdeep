library(gridExtra)
library(grid)
library(ggplot2)
library(dplyr)

plot_pcna = function(track, out_dir, prefix, minLength){
  print(paste("Plotting track #",prefix))
  MIN_LENGTH = minLength
  prefix = file.path(out_dir, prefix)
  # detect break point
  isBreakPoint = c(F)
  position = c(NaN)
  bin = T
  for (i in 2:nrow(track)){
    isBreakPoint = c(isBreakPoint, 
                     track$predicted_class[i-1] != track$predicted_class[i])
  }
  track["isBreakPoint"] = isBreakPoint
  for (i in 2:nrow(track)){
    if (track$isBreakPoint[i]==T){
      if (bin){ bin = F; position = c(position, T)} else { bin = T; position = c(position, F)}
    } else {position = c(position, NaN)}
  }
  track["isBreakPoint"] = isBreakPoint
  track["position"] = position
  
  # plot mitotic tracks
  r = vector()
  for(i in unique(track$lineageId)){
    if (length(unique(track$trackId[which(track$lineageId==i)]))>1){
      r = c(r, i)
    }
  }
  if (length(r) > 0){
    subtrack = subset(track, track$lineageId %in% r)
    subtrack$trackId = as.character(subtrack$trackId)
    subtrack$parentTrackId = as.factor(subtrack$parentTrackId)
    
    plots = vector('list',length(r))
    cbPalette = c("M" = "#FF0000", "G1/G2"="#696969", "S"="#0000CD")
    for (i in 1:length(r)){
      subtrack_new = subset(subtrack,subtrack$lineageId==r[i])
      p = ggplot(subtrack_new, aes(x=frame, y=trackId,color=predicted_class)) +
        geom_point(size=0.5) +
        theme_classic() +
        geom_text(
          data = subtrack_new %>% filter(isBreakPoint==TRUE & position == 0), 
          aes(label = frame), nudge_x = 0, nudge_y = 0.2, angle = 0, size = 3, check_overlap = T) + 
        geom_text(
          data = subtrack_new %>% filter(isBreakPoint==TRUE & position == 1), 
          aes(label = frame), nudge_x = 0, nudge_y = -0.2, angle = 0, size = 3, check_overlap = T) +
        scale_colour_manual(values=cbPalette) +
        labs(title=r[i]) +
        theme(legend.position = 'none', axis.title.x = element_blank(), axis.title.y=element_blank())
      plots[[i]] = p
    }
    
    rowOrg = 5
    colOrg = 4
    all = rowOrg * colOrg
    organized_plots = list()
    for(i in 1:ceiling(length(plots)/all)){
      organized_plots[[i]] = list()
    }
    for(i in 1:length(plots)){
      idx = i%%all
      if(idx==0){idx=all}
      organized_plots[[ceiling(i/all)]][[idx]] = plots[[i]]
    }
    ml = organized_plots %>% lapply(function(list) grid.arrange(grobs=list, nrow=rowOrg, ncol=colOrg))
    marrangeGrob(grobs=ml, ncol = 1, nrow = 1, top="Mitotis tracks", left="Lineages") %>%
      ggsave(paste(prefix,"-MitotisTracks.pdf",sep = ""),plot=.,device="pdf",units = "in", dpi = 300, width=12, height = 8)
  }
  
  
  # plot non-mitotic tracks with filtered length
  filtered_track = track
  filtered_track = filtered_track[c(),]
  for (i in unique(track$trackId)){
    p = track$frame[track$trackId==i]
    lg = unique(track$lineageId[track$trackId==i])
    if (length(lg)>1){
      print(paste("Warning! One track involved in two lineages. Check track ID:", i))
    } else {
      if (!length(p)==0 & !lg %in% r){
        if ((max(p) - min(p))>MIN_LENGTH){
          # select track longer than certain frames
          filtered_track = rbind(filtered_track, track[track$trackId==i,])
        }
      }
    }
  }
  filtered_track$trackId = as.character(filtered_track$trackId)
  count = unique(filtered_track$lineageId)
  plots = vector('list',length(count))
  rowNum = ceiling(length(plots)/4)
  cbPalette = c("M" = "#FF0000", "G1/G2"="#696969", "S"="#0000CD")
  
  for (i in 1:length(count)){
    filtered_track_new = subset(filtered_track, filtered_track$lineageId==count[i])
    p = ggplot(filtered_track_new, aes(x=frame, y=trackId,color=predicted_class)) +
      geom_point(size=0.5) +
      geom_text(
        data = filtered_track_new %>% filter(isBreakPoint==TRUE & position == 0), 
        aes(label = frame), nudge_x = 0, nudge_y = 0.2, angle = 0, size = 3, check_overlap = T) + 
      geom_text(
        data = filtered_track_new %>% filter(isBreakPoint==TRUE & position == 1), 
        aes(label = frame), nudge_x = 0, nudge_y = -0.2, angle = 0, size = 3, check_overlap = T) +
      theme_classic() + 
      scale_colour_manual(values=cbPalette) +
      labs(title=count[i]) +
      theme(legend.position = 'none', axis.title.x = element_blank(), axis.title.y=element_blank())
    plots[[i]] = p
  }
  
  rowOrg = 7
  colOrg = 4
  all = rowOrg * colOrg
  organized_plots = list()
  for(i in 1:ceiling(length(plots)/all)){
    organized_plots[[i]] = list()
  }
  for(i in 1:length(plots)){
    idx = i%%all
    if(idx==0){idx=all}
    organized_plots[[ceiling(i/all)]][[idx]] = plots[[i]]
  }
  ml = organized_plots %>% lapply(function(list) grid.arrange(grobs=list, nrow=rowOrg, ncol=colOrg))
  marrangeGrob(grobs=ml, ncol = 1, nrow = 1, top="Non-Mitotis tracks", left="Lineages") %>%
    ggsave(paste(prefix,"-NonMitotisTracks.pdf",sep = ""),plot=.,device="pdf",units = "in", dpi = 300, width=12, height = 8)
  
  lf = list.files(getwd(),pattern = "Rplots")
  file.remove(lf)
  print("Plottings saved.")
}
