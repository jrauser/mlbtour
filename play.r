library(tidyverse)

great_circle_distance <- function(lat1, lon1, lat2, lon2, radius = 6371) {
  # Convert degrees to radians
  to_rad <- function(deg) deg * pi / 180
  lat1 <- to_rad(lat1)
  lon1 <- to_rad(lon1)
  lat2 <- to_rad(lat2)
  lon2 <- to_rad(lon2)
  
  # Haversine formula
  delta_lat <- lat2 - lat1
  delta_lon <- lon2 - lon1
  
  a <- sin(delta_lat / 2)^2 + cos(lat1) * cos(lat2) * sin(delta_lon / 2)^2
  c <- 2 * atan2(sqrt(a), sqrt(1 - a))
  
  distance <- radius * c
  return(distance)  # in kilometers by default
}

haversine_dist <- function(lat1, lon1, lat2, lon2, radius = 6371) {
  to_rad <- function(deg) deg * pi / 180
  lat1 <- to_rad(lat1)
  lon1 <- to_rad(lon1)
  lat2 <- to_rad(lat2)
  lon2 <- to_rad(lon2)
  
  delta_lat <- outer(lat2, lat1, "-")  # rows: lat2, cols: lat1
  delta_lon <- outer(lon2, lon1, "-")
  
  lat1_rad <- matrix(to_rad(lat1), nrow = length(lat2), ncol = length(lat1), byrow = TRUE)
  lat2_rad <- matrix(to_rad(lat2), nrow = length(lat2), ncol = length(lat1), byrow = FALSE)
  
  a <- sin(delta_lat / 2)^2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2)^2
  c <- 2 * atan2(sqrt(a), sqrt(1 - a))
  
  radius * c  # distance matrix
}

venues <- read_csv("venues.csv", skip=1,
                   col_name=c("sport", "venue", "teams", "abbreviation", "lat", "long")) %>%
  filter(sport=="MLB") %>%
  select(-sport)
venues

dist_matrix <- haversine_dist(
  lat1 = venues$lat,
  lon1 = venues$long,
  lat2 = venues$lat,
  lon2 = venues$long
)

rownames(dist_matrix) <- venues$abbreviation
colnames(dist_matrix) <- venues$abbreviation

dist_matrix

?hclust

clusters<-hclust(as.dist(dist_matrix))
plot(clusters)


all_pairs <- crossing(origin=venues$abbreviation, destination=venues$abbreviation) %>%
  filter(origin != destination) %>%
  inner_join(venues %>% select(abbreviation, lat, long),
             by=join_by(origin == abbreviation)) %>%
  rename(origin_lat = lat, origin_long = long) %>%
  inner_join(venues %>% select(abbreviation, lat, long),
           by=join_by(destination == abbreviation)) %>%
  rename(dest_lat = lat, dest_long = long) %>%
  mutate(dist_km = great_circle_distance(origin_lat, origin_long, dest_lat, dest_long),
         dist_mi = dist_km / 1.609)

all_pairs %>% filter(destination > origin) %>% arrange(dist_mi)

all_pairs %>% filter(destination > origin) %>% arrange(dist_mi) %>%
  write_csv("all_pairs_distance.csv")
