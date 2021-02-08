# read params
# -t: track files (input repository)
# -c: "foci" files (cell cycle classification, input repository, should be different from tracks)
# -o: output repository
#   params for refinement
# -d: distance_tolerance (defalut: 40)
# -t: frame_tolerance (default: 5)
# -i: div_trans_factor (default: 2.5)
#   params for plotting
# -m: mininum length for a track to be plotted
library(getopt)
command=matrix(c( "track" , "t" ,1, "character" , "File path to raw track output",
                  "classification" , "c" ,2, "character" , "File path to cell cycle stage classification output",
                  "dist" , "d" ,2, "character" , "Distance tolerance (default: 40)",
                  "factor" , "i" ,2, "character" , "dist_trans_factor (default: 2.5)",
                  "frame" , "f" ,2, "character" , "Frame tolerance (default: 5)",
                  "out_dir", "o", 1, "character", "Output directory",
                  "minPlot", "m", 2, "character", "mininum length of a plotting",
                  "smooth" , "s" ,2, "integer" , "Smooth filter size (recommand: 5)",
                  "help", "h",0, "logical","Print usage"),byrow=T,ncol=5)
args=getopt(command)
if  (! is.null(args$help)) {
  cat(paste(getopt(command, usage = T),  "\n" ))
  q()
}
library(dplyr, warn.conflicts = F)
library(gridExtra, warn.conflicts = F)
library(grid, warn.conflicts = F)
library(ggplot2, warn.conflicts = F)

t = args$track
track = list.files(path=t, full.names = T, pattern = "*.csv")
if (length(track)==0){
  print("Error! No track found.")
  quit()
}
m = as.numeric(args$minPlot)
out_dir = args$out_dir
if (length(m)==0) { m = 50 }
warnings("off")
c = args$classification
if (length(c)==0){
  print("Error! Classification data missing.")
  quit()
}

distance_tolerance = as.numeric(args$dist)
if (length(distance_tolerance)==0) { distance_tolerance = 40 }
dist_factor = as.numeric(args$factor)
if (length(dist_factor)==0) { dist_factor = 2.5 }
frame_tolerance = as.numeric(args$frame)
if (length(frame_tolerance)==0) { frame_tolerance = 5 }
window_length = as.numeric(args$smooth)
if (length(window_length)==0) { window_length = 5 }

class = list.files(path=c, full.names = T, pattern = "*.csv")
# read all track .csv files in -t repository, stepwise, exit if broken.
# read all classificaiton .csv files in -c repository, stepwise, exit if broken.
if (length(unique(c(length(track),length(class))))!=1){
  print("Error: File number in repositories do not match!")
  quit()
}

merged_tracks = list()
prefix = vector()
# execute MergeTrackAndFoci.R, store [output1] .csv files in temp repository (files named in order, e.g. 1.csv, 2.csv ...)
source(file.path(getwd(), "MergeTrackAndFoci.R"))
for (i in 1:length(track)){
  prefix[i] = gsub('.csv','',basename(track[i]))
  merged_tracks[[i]] = mergeTrackAndFoci(read.csv(track[i]), read.csv(class[i]))
}

# read output file list in temp repository, for each file, run TrackRefine.R, store [output2] in temp repository.
# delete [output1] in temp
refined_tracks = list()
source(file.path(getwd(), "TrackRefine.R"))
for (i in 1:length(track)){
  print(paste("Refining tracks:",prefix[i]))
  refined_tracks[[i]] = trackRefine(merged_tracks[[i]], distance_tolerance, dist_factor, frame_tolerance, window_length)
  fp = file.path(out_dir, paste(prefix[i], '-refined.csv', sep=''))
  write.csv(refined_tracks[[i]], fp)
  print(paste("Refined tracks saved at:", fp))
  print('##============================================================')
}
unlink(merged_tracks)

# run Plotting.R (inspection mode) for each [output3] in output repository, store [output4] in the same repository.
source(file.path(getwd(), "Plotting.R"))
source(file.path(getwd(), "ResolveClass.R"))
phase = data.frame()
for (i in 1:length(track)){
  if (m > max(refined_tracks[[i]]$frame)){
    print("Error: minimum plotting track length larger than total frames.")
    exit()
  }
  plot_pcna(refined_tracks[[i]], out_dir, prefix[i], m)
  print('##=====================Resolving Class=========================')
  s = doResolveTrack(refined_tracks[[i]])
  s = cbind('stage'=rep(prefix[i],nrow(s)), s)
  print(paste("Resolved ",as.character(nrow(s)),"tracks."))
  phase = rbind(phase, s)
  print('##============================================================')
}
colnames(phase) = c('stage','lineage','type','G1','S','M','G2')
write.csv(phase, file.path(out_dir, 'phase.csv'), row.names = F)
unlink(file.path(out_dir, "Rplots.pdf"))
quit()