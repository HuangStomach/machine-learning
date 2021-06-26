library(Hmisc)
myvars <- c("mpg", "hp", "wt")
print(describe(mtcars[myvars]))
