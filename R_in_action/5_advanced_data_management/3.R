library(MASS)
options(digits = 3)
set.seed(1234)

mean <- c(230.7, 146.7, 3.6)
sigma <- matrix(c(
    15360.8, 6721.2, -47.1,
    6721.2, 4700.9, -16.5,
    -47.1, -16.5, 0.3
), nrow = 3, ncol = 3)
mydata <- mvrnorm(500, mean, sigma)
mydata <- as.data.frame(mydata)
names(mydata) <- c("y", "x1", "x2")
dim(mydata)
print(head(mydata, n = 10))
