mydata <- matrix(rnorm(30), nrow = 6)
apply(mydata, 1, mean)
apply(mydata, 2, mean)
apply(mydata, 2, mean, trim = 0.2)