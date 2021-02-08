resolve_phase = function(track, base=0, end=288){
  BASE=base
  END=end
  # Track: dataframe of one lineage, should have these fields
  #   lineageID: unique identifier of the lineage
  #   trackID: unique identifier of the track
  #   frame: time axis
  #   predicted_class: cell cycle identity
  transition = data.frame('frame'=vector(), 'trans'=vector())
  count_lineage = length(unique(track$trackId))
  track = track[order(track$frame, track$trackId),]
  if(count_lineage>3){return(NULL)}
  if(count_lineage==3){
    
    lineages = unique(track$trackId)
    m_entry = track$frame[min(which(track$predicted_class=='M'))]
    parent_lg = track$trackId[which(track$frame==m_entry)][1]
    parent = subset(track, track$trackId==parent_lg)
    
    if(parent$predicted_class[nrow(parent)]!='M'){return(NULL)}
    lineages = subset(lineages, lineages!=parent_lg)
    daug1_lg = lineages[1]
    daug2_lg = lineages[2]
    daug1 = subset(track, track$trackId==daug1_lg)
    daug2 = subset(track, track$trackId==daug2_lg)
    if(daug1$predicted_class[1]!='M' | daug2$predicted_class[1]!='M'){return(NULL)}
    
    # if parent have multiple M entry, correct the entry point by searching
    cd = parent$frame[which(parent$predicted_class=='M')]
    pin=1
    while((daug1$frame[min(which(daug1$predicted_class=='M'))]-m_entry)>10 |
          (daug2$frame[min(which(daug2$predicted_class=='M'))]-m_entry)>10){
      m_entry = cd[pin+1]
      pin = pin+1
    }
    
    out_parent = resolve_phase(parent, base=BASE, end=as.numeric(max(parent$frame)))
    out_daug1 = resolve_phase(daug1, base=m_entry, end=as.numeric(max(daug1$frame)))
    out_daug2 = resolve_phase(daug2, base=m_entry, end=as.numeric(max(daug2$frame)))
    if(is.null(out_parent) | is.null(out_daug1) | is.null(out_daug2)){return(NULL)}
    
    out = list()
    g1 = c(out_parent['G1'][[1]], out_daug1['G1'][[1]], out_daug2['G1'][[1]])
    g2 = c(out_parent['G2'][[1]], out_daug1['G2'][[1]], out_daug2['G2'][[1]])
    s = c(out_parent['S'][[1]], out_daug1['S'][[1]], out_daug2['S'][[1]])
    if(!is.null(g1)){
      out['G1'][[1]] = g1
    } else {out['G1']=NA}
    if(!is.null(g1)){
      out['G2'][[1]] = g2
    } else {out['G2']=NA}
    if(!is.null(s)){
      out['S'][[1]] = s
    } else {out['S']=NA}
    
    for(m_dur in 1:length(out_parent['M'][[1]])){
      # remove mitosis that is not involved in this relationship
      if(length(grep('>',out_parent['M'][[1]][m_dur]))>0){
        out_parent['M'][[1]] = out_parent['M'][[1]][-m_dur]
      }
    }
    out['M'][[1]] = mean(c(as.numeric(gsub('>','',out_daug1['M'][[1]][1])), 
                         as.numeric(gsub('>','',out_daug2['M'][[1]][1]))))
    return(out)
  }
  if(count_lineage==2){
    lineages = unique(track$trackId)
    if(!'M'%in%track$predicted_class){
      print(lineages)
      return(NULL)
    }
    m_entry = track$frame[min(which(track$predicted_class=='M'))]
    parent_lg = track$trackId[which(track$frame==m_entry)][1]
    daughter_lg = lineages[which(lineages!=parent_lg)]
    parent = subset(track, track$trackId==parent_lg)
    daughter = subset(track, track$trackId==daughter_lg)
    if(daughter$predicted_class[1]!='M'){return(NULL)}
    
    # if parent have multiple M entry, correct the entry point by searching
    cd = parent$frame[which(parent$predicted_class=='M')]
    pin=1
    while((daughter$frame[min(which(daughter$predicted_class=='M'))]-m_entry)>10){
      m_entry = cd[pin+1]
      pin = pin+1
    }

    out_parent = resolve_phase(parent)
    out_daughter = resolve_phase(daughter, base=m_entry)
    if(is.null(out_parent) | is.null(out_daughter)){return(NULL)}
    
    # organize track info
    out = list()
    g1 = c(out_parent['G1'][[1]], out_daughter['G1'][[1]])
    g2 = c(out_parent['G2'][[1]], out_daughter['G2'][[1]])
    s = c(out_parent['S'][[1]], out_daughter['S'][[1]])
    if(!is.null(g1)){
      out['G1'][[1]] = g1
    } else {out['G1']=NA}
    if(!is.null(g2)){
      out['G2'][[1]] = g2
    } else {out['G2']=NA}
    if(!is.null(s)){
      out['S'][[1]] = s
    } else {out['S']=NA}
    
    for(m_dur in 1:length(out_parent['M'][[1]])){
      # remove mitosis that is not involved in this relationship
      if(length(grep('>',out_parent['M'][[1]][m_dur]))>0){
        out_parent['M'][[1]] = out_parent['M'][[1]][-m_dur]
      }
    }
    out['M'][[1]] = mean(c(as.numeric(out_parent['M'][[1]][1]), 
                         as.numeric(gsub('>','',out_daughter['M'][[1]][1])))) # only handle one mitosis
    return(out)
  
  }else{
    
    # register state
    cur_state = track$predicted_class[1]
    trs_track = rbind(transition, c(track$frame[1],cur_state))
    colnames(trs_track)=c('frame','trans')
    for(i in 2:nrow(track)){
      if(track$predicted_class[i]!=track$predicted_class[i-1]){
        cur_state = track$predicted_class[i]
        trs_track = rbind(trs_track, c(track$frame[i], 
            paste(track$predicted_class[i-1],cur_state, sep='->')))
      }
    }
    trs_track = rbind(trs_track, c(track$frame[nrow(track)], cur_state))
    trs_track$frame = as.numeric(trs_track$frame)
    invalid = trs_track[which(trs_track$trans!='G1/G2->S' & trs_track$trans!='S->G1/G2' & 
                      trs_track$trans!='G1/G2->M' & trs_track$trans!='M->G1/G2' & 
                      !trs_track$trans %in% c('M','G1/G2','S')),]
    if(nrow(invalid)!=0){return(NULL)}
    # deduce G1/G2
    trs_track[which(trs_track$trans=='G1/G2->S'),'trans'] = 'G1->S'
    trs_track[which(trs_track$trans=='G1/G2->M'),'trans'] = 'G2->M'
    trs_track[which(trs_track$trans=='M->G1/G2'),'trans'] = 'M->G1'
    trs_track[which(trs_track$trans=='S->G1/G2'),'trans'] = 'S->G2'
    
    # deduce duration
    if(nrow(trs_track)==2){
      # arrest
      out = list()
      out[trs_track$trans[1]] = 'arrest'
      return(out)
    }
    if(nrow(trs_track)==3){
      # single transition
      out = list()
      phs = strsplit(trs_track$trans[2],'->')[[1]]
      out[phs[1]] = paste('>',trs_track$frame[2]-BASE,sep='')
      out[phs[2]] = paste('>',END-trs_track$frame[2],sep='')
      return(out)
    }
    out = list()
    phs = strsplit(trs_track$trans[2],'->')[[1]]
    out[phs[1]]=paste('>',trs_track$frame[2]-BASE,sep='')
    phs = strsplit(trs_track$trans[nrow(trs_track)-1],'->')[[1]]
    out[phs[2]]=paste('>',END-trs_track$frame[nrow(trs_track)],sep='')
    for(i in 3:(nrow(trs_track)-1)){
      prv_state = strsplit(trs_track$trans[i-1],'->')[[1]][2]
      trs_state = strsplit(trs_track$trans[i],'->')[[1]][1]
      if(prv_state!=trs_state){return(NULL)}
      if(trs_state %in% names(out)){
        out[trs_state][[1]] = c(out[trs_state][[1]], trs_track$frame[i]-trs_track$frame[i-1])
      }else{
        out[trs_state][[1]] = trs_track$frame[i]-trs_track$frame[i-1]
      }
    }
    return(out)
  }
}

doResolveTrack = function(track, length_filter=200){
  lineage_count = length(unique(track$lineageId))
  out = data.frame('lineage'=unique(track$lineageId),
                   'type'=rep(NA, lineage_count),
                   'G1'=rep(NA,lineage_count),
                   'S'=rep(NA,lineage_count),
                   'M'=rep(NA,lineage_count),
                   'G2'=rep(NA,lineage_count))
  for(i in unique(track$lineageId)){
    d = subset(track, track$lineageId==i)
    if(length(unique(d$trackId))==1 & (max(d$frame)-min(d$frame))<length_filter){next}
    idx = which(out$lineage==i)
    rsd = resolve_phase(d, base=as.numeric(min(d$frame)), end=as.numeric(max(d$frame)))
    if(is.null(rsd)){next}
    if(length(rsd)==1){
      out$type[idx] = paste(names(rsd),'arrest',max(track$frame)-min(track$frame),sep='_')
    } else {
      out$type[idx] = 'non-mitosis'
      if(!is.null(rsd$S[[1]])){
        s = rsd$S[[1]]
        for(dur in s){
          if(length(grep('[><]',dur))==0){
            if(is.na(out$S[idx])){
              out$S[idx] = as.numeric(s)
            } else {
              out$S[idx] = paste(out$S[idx],s,sep='/')
            }
          }
        }
      }
      if(!is.null(rsd$G1[[1]])){
        g1 = rsd$G1[[1]]
        for(dur in g1){
          if(length(grep('[><]',dur))==0){
            if(is.na(out$G1[idx])){
              out$G1[idx] = as.numeric(g1)
            } else {
              out$G1[idx] = paste(out$G1[idx],g1,sep='/')
            }
          }
        }
      }
      if(!is.null(rsd$M[[1]])){
        if(length(unique(d$trackId))==1){
          out$type[idx] = 'mitosis_lose_daughter'
        } else {
          out$type[idx] = 'mitosis'
        }
        m = rsd$M[[1]]
        for(dur in m){
          if(length(grep('[><]',dur))==0){
            if(is.na(out$M[idx])){
              out$M[idx] = as.numeric(m)
            } else {
              out$M[idx] = paste(out$M[idx],m,sep='/')
            }
          }
        }
      }
      if(!is.null(rsd$G2[[1]])){
        g2 = rsd$G2[[1]]
        for(dur in g2){
          if(length(grep('[><]',dur))==0){
            # filter out tracks with single transition (>xxx, <xxx frame)
            if(is.na(out$G2[idx])){
              out$G2[idx] = as.numeric(g2)
            } else {
              out$G2[idx] = paste(out$G2[idx],g2,sep='/')
            }
          }
        }
      }
      if(is.na(out$G1[idx]) & is.na(out$S[idx]) & is.na(out$G2[idx]) & is.na(out$M[idx])){
        # filter out tracks with single transition
        out$type[idx] = NA
      }
    }
  }
  out = subset(out, !is.na(out$type))
  return(out)
}
