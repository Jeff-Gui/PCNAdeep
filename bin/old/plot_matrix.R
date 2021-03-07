library(tidyverse)
library(gridExtra)

nv_trans = function(x){
  # transform x string to name value pair vectur
  x = gsub("[{}(\\\") ]", "", x)
  return(strsplit(x, ':')[[1]])
}

process_matrix = function(fp){
  #fname = '/Users/jefft/Desktop/Chan lab/SRTP/ImageAnalysis/PCNAdeep/train_matrix/20210125_kaggle.json'
  fname = fp
  data = readChar(fname, nchars = file.info(fname)$size)
  data_lines = strsplit(data, '\n')[[1]]
  
  AP = data.frame()
  meta = data.frame('data_time'=vector(), 'eta_seconds'=vector(), 'fast_rcnn/cls_accuracy'=vector(),
                    'fast_rcnn/false_negative'=vector(), 'fast_rcnn/fg_cls_accuracy'=vector(),
                    'iteration'=vector(), 'loss_box_reg'=vector(), 'loss_cls'=vector(), 'loss_mask'=vector(),
                    'loss_rpn_cls'=vector(), 'loss_rpn_loc'=vector(), 'lr'=vector(),
                    'mask_rcnn/accuracy'=vector(), 'mask_rcnn/false_negative'=vector(),
                    'mask_rcnn/false_positive'=vector(), 'roi_head/num_bg_samples'=vector(),
                    'roi_head/num_fg_samples'=vector(), 'rpn/num_neg_anchors'=vector(),
                    'rpn/num_pos_anchors'=vector(), 'time'=vector(), 'total_loss'=vector())
  
  for (line in data_lines){
    fields = strsplit(line, ',')[[1]]
    nms = c()
    values = c()
    i = 1
    while (i<=length(fields)){
      v = nv_trans(fields[i])
      if (length(grep('segm', v[1]))>0){
        AP_row = c(values[6])
        while(length(grep('segm', nv_trans(fields[i])))>0){
          AP_row = c(AP_row, fields[i])
          i = i+1
        }
        AP = rbind(AP, AP_row)
        next
      }
      nms = c(nms, v[1])
      values = c(values, as.numeric(v[2]))
      i = i+1
    }
    names(values) = nms
    meta[nrow(meta)+1,] = values
  }
  
  meta = meta[1:nrow(meta)-1,]
  meta$iteration = as.numeric(meta$iteration)
  
  AP = AP[1:nrow(AP)-1,]
  AP_name = c('iteration')
  for (i in 2:length(names(AP))){
    AP_name = c(AP_name, gsub('segm.', '',strsplit(names(AP)[i],'\\.\\.\\.\\.')[[1]][2]))
  }
  colnames(AP) = AP_name
  for (i in 1:nrow(AP)){
    for (j in 1:ncol(AP)){
      AP[i,j] = gsub('\\S*: ','',AP[i,j])
    }
  }
 
  #AP = AP[, which(colnames(AP)!='APl')]
  AP = gather(AP, key='AP_measures', value='value', 2:ncol(AP))
  AP$value = as.numeric(AP$value)
  AP$iteration = as.numeric(AP$iteration)
  return(list('AP'=AP, 'meta'=meta))
}

plot_matrix = function(IMAGE_TRAIN, BATCH_SIZE, MAX_ITER, AP, meta, bs_factor=1){
  BATCH_SIZE = BATCH_SIZE * bs_factor
  iter_per_epoch = floor(IMAGE_TRAIN/BATCH_SIZE)
  
  cut = c()
  ep = iter_per_epoch
  while (ep <= MAX_ITER){
    cut = c(cut, ep)
    ep = ep + iter_per_epoch
  }
  
  ap = ggplot(AP, aes(x=iteration,y=value, color=AP_measures)) + theme_classic() +
    geom_line() + 
    geom_vline(xintercept=cut, linetype="dotted") +
    labs(x='Iteration', y='%', title = 'Average precision') +
    theme(legend.position = 'bottom', legend.title=element_blank(), 
          legend.box.spacing = unit(0,'cm'), legend.key.size = unit(3,'mm'))
  
  lc = ggplot(meta) + theme_classic() + 
    geom_line(aes(x=iteration, y=total_loss)) + 
    geom_vline(xintercept=cut, linetype="dotted") +
    labs(x='Iteration', y=element_blank(),title='Total loss')
  
  lr = ggplot(meta) + theme_classic() +
    geom_line(aes(x=iteration, y=lr)) +
    geom_vline(xintercept=cut, linetype="dotted") +
    labs(x='Iteration', y=element_blank(),title = 'Learning rate')
  
  return(list('ap'=ap, 'lc'=lc, 'lr'=lr))
}



