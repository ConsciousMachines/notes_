





# a)

library(igraph)

zachary_dir = '/home/chad/Desktop/skool/840/zachary.txt'
data = read.table(zachary_dir)
data = data[,c(1,2)] # remove weights column
g = graph_from_edgelist(as.matrix(data), directed=FALSE)

# draw the graph, find diameter of graph
# https://r.igraph.org/reference/plot.common.html?q=draw#null
plot.igraph(g)
# https://r.igraph.org/reference/diameter.html?q=diameter#null
diameter(g, directed = FALSE)




# b)
# find the degree distribution
# https://r.igraph.org/reference/degree.html?q=degree%20distrib#null 
plot(degree_distribution(g, cumulative=FALSE), type='l', main='degree distribution')






# c)
# find clustering coefficients of all vertices and for the graph
# https://r.igraph.org/reference/transitivity.html?q=clustering%20coeff#null
# number 12 is NAN because it only has one neighbor
# clustering coefficient for all vertices
plot(transitivity(g, type="local"), type='l', main='clustering coefficient for all vertices')
# clustering coefficient for entire graph
transitivity(g, type = "global")






# d)
# find closeness and betweenness centralities of the vertices
# https://r.igraph.org/reference/closeness.html?q=closeness#null
plot(closeness(g), type='l', main='closeness')
# https://r.igraph.org/reference/betweenness.html?q=betweenness#null
plot(betweenness(g), type='l', main='betweenness')

