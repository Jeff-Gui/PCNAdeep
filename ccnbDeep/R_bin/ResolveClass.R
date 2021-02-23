resolve_phase = function(track, base=0, end=288, s_min=10){
  BASE=base
  END=end
  S_MIN=s_min
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
    if(!'M'%in%track$predicted_class){return(NULL)}
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
      if(is.na(m_entry)){return(NULL)}
      pin = pin+1
    }
    
    out_parent = resolve_phase(parent, base=BASE, end=as.numeric(max(parent$frame)), S_MIN)
    trans_par = out_parent$transition
    out_parent = out_parent$out
    out_daug1 = resolve_phase(daug1, base=m_entry, end=as.numeric(max(daug1$frame)), S_MIN)
    trans_daug1 = out_daug1$transition
    out_daug1 = out_daug1$out
    out_daug2 = resolve_phase(daug2, base=m_entry, end=as.numeric(max(daug2$frame)), S_MIN)
    trans_daug2 = out_daug2$transition
    out_daug2 = out_daug2$out
    
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
    return(list('out'=out,'transition'=list('parent'=trans_par, 'daug1'=trans_daug1, 'daug2'=trans_daug2)))
  }
  if(count_lineage==2){
    lineages = unique(track$trackId)
    if(!'M'%in%track$predicted_class){return(NULL)}
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
      if(is.na(m_entry)){return(NULL)}
      pin = pin+1
    }

    out_parent = resolve_phase(parent, base=BASE, end=as.numeric(max(parent$frame)), S_MIN)
    trans_par = out_parent$transition
    out_parent = out_parent$out
    out_daughter = resolve_phase(daughter, base=m_entry, end=as.numeric(max(daughter$frame)), S_MIN)
    trans_daug = out_daughter$transition
    out_daughter = out_daughter$out
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
    if (out_daughter['M'][[1]][1]=='arrest'){
      out['M'][[1]] = as.numeric(out_parent['M'][[1]][1])
    } else {
    out['M'][[1]] = mean(c(as.numeric(out_parent['M'][[1]][1]), 
                         as.numeric(gsub('>','',out_daughter['M'][[1]][1])))) # only handle one mitosis
    }
    
    return(list('out'=out,'transition'=list('parent'=trans_par, 'daug'=trans_daug)))
    
  }else{
    flag = F
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
    for (i in which(trs_track$trans=='G1/G2->S')){
      if (trs_track$trans[i+1]=='S->G1/G2'){
        if (trs_track$frame[i+1]-trs_track$frame[i]<S_MIN){
          track$predicted_class[which(track$frame==trs_track$frame[i]):
                                which(track$frame==trs_track$frame[i+1])] = 'G1/G2'
          print(paste('omitted short S:',trs_track$frame[i+1]-trs_track$frame[i]))
          flag = T
        }
      }
    }
    for (i in which(trs_track$trans=='S->G1/G2')){
      if (trs_track$trans[i+1]=='G1/G2->S'){
        if (trs_track$frame[i+1]-trs_track$frame[i]<S_MIN){
          track$predicted_class[which(track$frame==trs_track$frame[i]):
                                  which(track$frame==trs_track$frame[i+1])] = 'G1/G2'
          print(paste('omitted short G1/G2:',trs_track$frame[i+1]-trs_track$frame[i]))
          flag = T
        }
      }
    }
    if (flag){return(resolve_phase(track, BASE, END, S_MIN))}
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
      return(list('out'=out, 'transition'=list('single'=trs_track)))
    }
    if(nrow(trs_track)==3){
      # single transition
      out = list()
      phs = strsplit(trs_track$trans[2],'->')[[1]]
      out[phs[1]] = paste('>',trs_track$frame[2]-BASE,sep='')
      out[phs[2]] = paste('>',END-trs_track$frame[2],sep='')
      return(list('out'=out, 'transition'=list('single'=trs_track)))
    }
    out = list()
    phs = strsplit(trs_track$trans[2],'->')[[1]]
    out[phs[1]]=paste('>',trs_track$frame[2]-BASE,sep='')
    phs = strsplit(trs_track$trans[nrow(trs_track)-1],'->')[[1]]
    out[phs[2]]=paste('>',END-trs_track$frame[nrow(trs_track)-1],sep='')
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
    return(list('out'=out, 'transition'=list('single'=trs_track)))
  }
}

extract_mitosis = function(track, length_filter=200, minM=5, prevM=20, postM=10){
  #  extract mitosis track through resolving the class, length filter 5 (minM)
  #  extract frame info before mitosis, max 20 (prevM) and after entry, max 10
  #  output: suggested mitosis track ID (parent track, not daughter track (?))
  lineage_count = length(unique(track$lineageId))
  out = data.frame()
  for(i in unique(track$lineageId)){
    d = subset(track, track$lineageId==i)
    if(length(unique(d$trackId))==1 & (max(d$frame)-min(d$frame))<length_filter){next}
    if(length(grep('M',d$predicted_class))==0){next}
    rsd = resolve_phase(d, base=as.numeric(min(d$frame)), end=as.numeric(max(d$frame)), s_min=10)
    if(is.null(rsd)){next}
    trans = rsd$transition
    if (length(trans)>=2){
      trans = trans[['parent']][[1]]
    } else {
      trans = trans[[1]]
    }
    if(length(grep('G2->M', trans$trans))==0){next}
    if(as.numeric(gsub('>','', rsd$out$M))>minM){
      daug = unique(d$parentTrackId)
      daug = daug[which(daug!=0)]
      if(length(daug)==0){
        out = rbind(out, c(unique(d$lineageId),trans$frame[which(trans$trans=='G2->M')], NA))
      } else {
        out = rbind(out, c(unique(d$lineageId),trans$frame[which(trans$trans=='G2->M')], paste(daug, collapse = '/')))
      }
    }
  }
  colnames(out) = c('trackId', 'M_entry', 'daughterTrackId')
  filtered_track = data.frame()
  for (i in 1:nrow(out)){
    id = out$trackId[i]
    entry = as.numeric(out$M_entry[i])
    sub = track[which(track$trackId==id & track$frame>=(entry-prevM) & track$frame<=(entry+postM)),]
    sub['is_entry'] = rep(0, nrow(sub))
    sub$is_entry[which(sub$frame==entry)] = 1
    filtered_track = rbind(filtered_track, sub)
  }
  return(list('meta'=out, 'track'=filtered_track))
}

