# mapa de calor

# datos de los incendios bajados del proyecto Kaggle "Fires from Space"
# disponible en https://www.kaggle.com/carlosparadis/fires-from-space-australia-and-new-zeland

# divisiones administrativas de Australia bajadas de DIVA-GIS
# disponible en http://www.diva-gis.org/datadown


#install.packages('ggmap')
#install.packages('sf')

# cargar paquetes
library(ggplot2)
#library(ggmap)
library(dplyr)
library(rgdal)
library(sp)
library(sf)
#install.packages('rmapshaper')
library(rmapshaper)
#install.packages('viridis')
#library(viridis)
#install.packages('lwgeom')
library(lwgeom)
install.packages('spData')
library(spData)

setwd("/Users/jordan.j.fischer@ibm.com/Documents/Training/RTraining/30diasdegraficos/mapa de calor")

fires1 <- read.csv('fire_archive_M6_96619.csv')
#fires2 <- read.csv('fire_nrt_M6_96619.csv')
#fires3 <- read.csv('fire_archive_V1_96617.csv')
#fires4 <- read.csv('fire_nrt_V1_96617.csv')




#adm1 <- readOGR("AUS_adm/AUS_adm1.shp")
adm2 <- readOGR("AUS_adm/AUS_adm2.shp")

#plot(adm1)
## check data already embedded in shapefile
head(adm2@data)

#library(raster)
#adm1 <- shapefile("AUS_adm/AUS_adm1.shp")
#head(adm1@data)

#map <- ggplot() + 
#  geom_polygon(data = adm1, aes(x = long, y = lat, group = group), colour = "black", fill = NA) + 
#  geom_point(data=fires1, aes(x=longitude, y=latitude, color=brightness, size = 0.5)) + scale_color_continuous_sequential(palette = "Heat")
#map

#map2 <- ggplot(fires1 %>%
#           arrange(brightness),
#         aes(x=longitude, y=latitude, color=brightness, size = 0.5)) +
#  geom_point() + scale_color_gradient(low="red", high="yellow") 
#map2



map3 <- ggplot() + 
#  geom_polygon(data = adm2, aes(x = long, y = lat, group = group), colour = "gray", fill = NA) + 
  geom_point(data = fires1 %>%
               arrange(brightness),
             aes(x=longitude, y=latitude, color=brightness)) + 
  scale_color_gradient(name = 'intensidad', low="yellow", high="red") +
  ggtitle("incendios en Australia verano 2019-2020") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme_void()
map3


#merge shp and csv

#vars <- c(fires1$latitude, fires1$longitude)


crs <- st_crs(adm2)  # get shapefile crs so points crs will match
#poly <- st_as_sf(adm1)
#object.size(poly)
# to minimize processing time, use st_simplify() (???)
#poly <- st_simplify(poly, dTolerance = 2000)
#poly$geometry = as.numeric(poly$geometry)
#poly_simp = rmapshaper::ms_simplify(poly, keep = 0.01,
#                                          keep_shapes = TRUE)
poly2 <- st_as_sf(adm2)

object.size(poly2)
poly2_simp = rmapshaper::ms_simplify(poly2, keep = 0.01,
                                     keep_shapes = TRUE)
object.size(poly2_simp)

ggplot() +
  geom_sf(data = poly2_simp) 

#length(unique(poly2_simp$NAME_2))
#length(unique(poly2_simp$ID_2))



pts <- st_as_sf(fires1, coords = c("longitude", "latitude"), crs = crs)
#st_crs(pts)    # double check crs


#joined <- st_join(pts, poly, join = st_intersects)
joined2 <- st_join(poly2_simp, pts, join = st_intersects)
head(joined2)
table(joined2$ID_2, exclude = NULL)

# the spatial join missed 11 points (out of 36011) they can be manually added or just dropped
#joined["8703", "NAME_1"] <- 'Northern Territory'
#joined_complete <- joined[!is.na(joined$NAME_1), ]

#table(joined_complete$NAME_1, exclude = NULL)



grouped <- joined2 %>%
  group_by(ID_2) %>%
  mutate(fire_by_adm = sum(brightness, na.rm = TRUE))

table(grouped$ID_2, exclude = NULL)

#grouped2 <- grouped %>%
#  select(NAME_1, geometry, fire_by_adm1)

per_territory <- grouped[!duplicated(grouped$ID_2),]

per_territory$area <- st_area(per_territory$geometry)/1000000   # area in 1000s of kmˆ2

#start to play with scales
min = min(per_territory$fire_by_adm)
max = max(per_territory$fire_by_adm)
diff <- max - min
std = sd(per_territory$fire_by_adm)


quantile.interval = quantile(per_territory$area, probs=seq(0, 1, by = 1/6))
std.interval = c(seq(min, max, by=std), max)

require(classInt)
natural.interval = classIntervals(per_territory$fire_by_adm, n = 10, style = 'jenks')$brks
#natural.interval = classIntervals(per_territory$fire_by_adm, n = 10, style = 'jenks')



per_territory$fire_by_adm.quantile = cut(per_territory$area, breaks=quantile.interval, include.lowest = TRUE)
#per_territory$fire_by_adm.quantile = as.numeric(per_territory$fire_by_adm.quantile)
per_territory$fire_by_adm.std = cut(per_territory$area, breaks=std.interval, include.lowest = TRUE)
per_territory$fire_by_adm.natural = cut(per_territory$area, breaks=natural.interval, include.lowest = TRUE)


cc <- scales::seq_gradient_pal("red", "yellow", "Lab")(seq(0,1,length.out=10))


mapa_calor <- ggplot() +
  geom_sf(data = per_territory, color='white', aes(fill=per_territory$fire_by_adm.natural), lwd = 0.1) + 
  scale_fill_manual(name = 'intensidad', values=cc) +
  ggtitle("adm2 de Australia por intensidad de incendios, verano 2019-2020") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme_void()
mapa_calor



mapa_calor <- ggplot() +
  geom_sf(data = per_territory, color='white', aes(fill=fire_by_adm), lwd = 0.3) + 
  scale_fill_gradient(name = 'intensidad', low="yellow", high="red") +
  ggtitle("adm2 de Australia por intensidad de incendios, verano 2019-2020") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme_void()
mapa_calor






#plot the result (this is computationally expensive... be prepared to wait a while depending on your processing power)
ggplot() +
  geom_sf(data = per_territory, color='white', aes(fill=fire_by_adm), lwd = 0.3) + 
  scale_fill_gradient(low="yellow", high="red") +
  theme_void()




#adm2 <- readOGR("AUS_adm/AUS_adm2.shp")




#points viz but with already joined and simplified data
map3 <- ggplot() + 
  geom_sf(data = poly2, colour = "gray", fill = NA) + 
  geom_sf(data = pts %>%
               arrange(brightness),
             aes(color=brightness)) + 
  scale_color_gradient(low="yellow", high="red") +
  theme_void()
map3







#
