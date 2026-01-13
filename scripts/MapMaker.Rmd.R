---
  title: "IN Ponds Map"
header-includes:
  - \usepackage{array}
output: pdf_document
latex_engine: xelatex
geometry: margin=2.54cm
--- 
  
  
  
```{r, results = 'hide'}

rm(list=ls())
getwd()
setwd("~/GitHub/micro_macro_perspective") 

#install.packages("readr")


require(sp)
require(gstat)
require(raster)
require(maptools)
require(maps)
require(mapdata)
require(rgdal)
require(rgeos)
require(rgdal)
require(dplyr)
require(ggplot2)
require(ggmap)
require(devtools)
require(stringr)
require(gridExtra)
require(grid)
require(rasterVis)
require(scales)
require(ggsn)
require(jpeg)
require(ripa)
require(cowplot)

sample.ponds <- read.table("~/GitHub/micro_macro_perspective/data/20130801_INPondDataMod.csv", sep = ",", header = TRUE)

all.ponds <- read.table("~/GitHub/DistDecay/map/RefugePonds.csv", sep = ",", header = TRUE)
```