myvars <- c("mpg", "hp", "wt")
aggregate(mtcars[myvars], by = list(am = mtcars$am), mean)
aggregate(mtcars[myvars], by = list(am = mtcars$am), sd)