states <- data.frame(state.region, state.x77)
means <- aggregate(states$Illiteracy, by = list(state.region), FUN = mean)
means <- means[order(means$x), ]
barplot(means$x, names.arg = means$Group.1)
title("Mean Illiteracy Rate")
