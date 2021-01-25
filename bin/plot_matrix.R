library(tidyverse)
fname = '/Users/jefft/Desktop/Chan lab/SRTP/ImageAnalysis/PCNAdeep/train_matrix/20210125_kaggle.json'
data = readChar(fname, nchars = file.info(fname)$size)
data_lines = strsplit(data, '\n')[[1]]

AP = data.frame('iteration'=vector(), 'AP'=vector(), 'AP50'=vector(), 
           'AP75'=vector(), 'APl'=vector(), 'APm'=vector(), 'APs'=vector())
meta = data.frame('data_time'=vector(), 'eta_seconds'=vector(), 'fast_rcnn/cls_accuracy'=vector(),
                  'fast_rcnn/false_negative'=vector(), 'fast_rcnn/fg_cls_accuracy'=vector(),
                  'iteration'=vector(), 'loss_box_reg'=vector(), 'loss_cls'=vector(), 'loss_mask'=vector(),
                  'loss_rpn_cls'=vector(), 'loss_rpn_loc'=vector(), 'lr'=vector(),
                  'mask_rcnn/accuracy'=vector(), 'mask_rcnn/false_negative'=vector(),
                  'mask_rcnn/false_positive'=vector(), 'roi_head/num_bg_samples'=vector(),
                  'roi_head/num_fg_samples'=vector(), 'rpn/num_neg_anchors'=vector(),
                  'rpn/num_pos_anchors'=vector(), 'time'=vector(), 'total_loss'=vector())

nv_trans = function(x){
  # transform x string to name value pair vectur
  x = gsub("[{}(\\\") ]", "", x)
  return(strsplit(x, ':')[[1]])
}

for (line in data_lines){
  fields = strsplit(line, ',')[[1]]
  nms = c()
  values = c()
  i = 1
  while (i<=length(fields)){
    v = nv_trans(fields[i])
    if (length(grep('segm', v[1]))>0){
      AP_row = c(values[6], v[2], nv_trans(fields[i+1])[2], 
                 nv_trans(fields[i+2])[2], nv_trans(fields[i+3])[2],
                 nv_trans(fields[i+4])[2], nv_trans(fields[i+5])[2])
      AP[nrow(AP)+1,] = AP_row
      i = i+6
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
AP = AP[, which(colnames(AP)!='APl')]
AP = gather(AP, key='AP_measures', value='value', 2:ncol(AP))
AP$value = as.numeric(AP$value)
AP$iteration = as.numeric(AP$iteration)

#======================= Plotting ==================================
ITER_PER_EPOCH = 285
MAX_ITER = 3000
cut = c()
ep = ITER_PER_EPOCH
while (ep <= MAX_ITER){
  cut = c(cut, ep)
  ep = ep + ITER_PER_EPOCH
}

ggplot(AP, aes(x=iteration,y=value, color=AP_measures)) + theme_classic() +
  geom_line() +
  geom_vline(xintercept=cut, linetype="dotted") +
  labs(x='Iteration', y='%', title = 'Average precision')

ggplot(meta) + theme_classic() + 
  geom_line(aes(x=iteration, y=total_loss)) + 
  geom_vline(xintercept=cut, linetype="dotted") +
  labs(x='Iteration', y='Total loss', title='Learning curve')

ggplot(meta) + theme_classic() +
  geom_line(aes(x=iteration, y=lr)) +
  geom_vline(xintercept=cut, linetype="dotted") +
  labs(x='Iteration', y='Learning rate', title = 'Learning rate')



