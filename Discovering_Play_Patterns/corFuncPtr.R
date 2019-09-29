
# 논문에서 구한 방식 : 너무 느리다... 
# library(TSclust)
# start_time <- Sys.time()
# d<- diss(COR, 'COR')
# end_time <- Sys.time()
# print( end_time - start_time )
# p = cor(COR[idx1,], COR[idx2,])
# sqrt(2 * (1 - p))


# 병렬처리 
# http://arma.sourceforge.net/docs.html

# RcppArmadillo is used as dependency
library(RcppArmadillo)
 # Use RcppXPtrUtils for simple usage of C++ external pointers
library(RcppXPtrUtils)
# compile user-defined function and return pointer (RcppArmadillo is used as dependency)


corFuncPtr <- cppXPtr("double customDist(const arma::mat &A, const arma::mat &B) {
                                 return sqrt( 2 * ( 1 - arma::accu(arma::cor(A , B)) )); }",
                            depends = c("RcppArmadillo"))
# distance matrix for user-defined euclidean distance function
 # (note that method is set to "custom")
