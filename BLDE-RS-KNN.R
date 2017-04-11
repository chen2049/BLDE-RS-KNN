# BLDE-RS-KNN classifier

# BLDE Algorithm
# population_size: Population size
# chromosome_size: Chromosome size
# max_iterate: Maximum number of iterations
# p_threshold: Mutation threshold
# function_validate: calculate accuracy function of validate set

BLDE_AG <- function(population_size, chromosome_size, max_iterate, p_threshold, function_validate)
{
  X <- matrix(sample(c(0, 1), population_size*chromosome_size, replace=T), nrow=population_size, ncol=chromosome_size)
  A <- matrix(sample(c(0, 1), population_size*chromosome_size, replace=T), nrow=population_size, ncol=chromosome_size)
  for(times in 1:max_iterate){
    middle_varible <- X
    accuracy <- apply(X, 1, function_validate)
    x_gb <- X[which(accuracy==max(accuracy))[1], ]
    for(w in 1:population_size){
      s <- sample(1:population_size, 2, replace=T)
      x <- X[s[1], ]
      y <- X[s[2], ]
      z <- A[sample(1:population_size, 1), ]
      if(function_validate(y) >= function_validate(z)){
        tx <- y
      }else{
        tx <- z
      }
      for(j in 1:chromosome_size){
        if(y[j]==z[j]){
          if(x_gb[j]!=x[j]){
            tx[j] <- x_gb[j]
          }else{
            if(runif(1, min=0, max=1)<=p_threshold){
              tx[j] <- sample(c(0, 1), 1)
            }
          }
        }
      }
      if(function_validate(tx) >= function_validate(X[w, ])){
        X[w, ] <- tx
      }
    }
    A <- middle_varible
  }
  return(x_gb)
}


# sub_dim: the dim of subspace
# nresample: times of resample
set.seed(10)
sub_col_index <- function(data_train, sub_dim, nresample){
  col_index <- matrix(sample(colnames(data_train[, -ncol(data_train)]), nresample*sub_dim, T), nresample, sub_dim, byrow=T)
  col_index <- cbind(rep('y', nresample), col_index)
  return(col_index)
}

# Classfier Pool
# data_train: train set
# data_validate: validate set
# data_test: test set
# nresample: times of resample
# K: the parameter of knn 

library(class)
class_pool <- function(data_train, data_validate, data_test, sub_col, nresample, K)
{
  train_result <- matrix(NA, nrow=nrow(data_train), ncol=nresample+1)
  validate_result <- matrix(NA, nrow=nrow(data_validate), ncol=nresample+1)
  test_result <- matrix(NA, nrow=nrow(data_test), ncol=nresample+1)
  for(j in 1:nresample){
    col_index <- sub_col[j, ]
    sub_train <- data_train[, col_index]
    sub_validate <- data_validate[, col_index]
    sub_test <- data_test[, col_index]
    pre_v <- knn(sub_train[, -1], sub_validate[, -1], cl = sub_train[, 1], k = K) 
    pre_t <- knn(sub_train[, -1], sub_test[, -1], cl = sub_train[, 1], k = K)
    validate_result[, j] <- as.numeric(as.matrix(pre_v))
    test_result[, j] <- as.numeric(as.matrix(pre_t))
  }
  return(rbind(train_result,validate_result, test_result))
}


# accuracy function of RS-KNN

validate_accuracy <- function(x)
{
	validate_result <- cbind(validate_result[, which(x==rep(1, length(x)))], validate_result[, length(x)+1])
	validate_result[, sum(x)+1] <- round(rowSums(validate_result[, -(sum(x)+1)])/sum(x))
	k <- sum(validate_result[, sum(x)+1]==data_validate[, 'y'])/nrow(data_validate)
	return(k)
}

# Overall accuracy: k 
# First type of precision: k_1
# Second type of precision: k_2

test_accuracy <- function(x)
{
	test_result <- cbind(test_result[, which(x==rep(1, length(x)))], test_result[, length(x)+1])
	test_result[, sum(x)+1] <- round(rowSums(test_result[, -(sum(x)+1)])/sum(x))
	k <- sum(test_result[, sum(x)+1]==data_test[, 'y'])/nrow(data_test)
	type_1_result <- test_result[data_test[, 'y'] == rep(0, nrow(data_test)), ]
	type_2_result <- test_result[data_test[, 'y'] == rep(1, nrow(data_test)), ]
	k_1 <- sum(type_1_result[, sum(x)+1] == rep(0, nrow(type_1_result)))/nrow(type_1_result)
	k_2 <- sum(type_2_result[, sum(x)+1] == rep(1, nrow(type_2_result)))/nrow(type_2_result)
    return(c(k, k_1, k_2))
}


# calculate auc statisics
library(pROC)
# validate auc
validate_auc <- function(x){
  validate_result <- cbind(validate_result[, which(x==rep(1, length(x)))], validate_result[, length(x)+1])
  pre_prob <- rowSums(validate_result[, -(sum(x)+1)])/sum(x)
  return(roc(data_validate[, 'y'], pre_prob)$auc + 0)
}

# test auc
test_auc <- function(x){
  test_result <- cbind(test_result[, which(x==rep(1, length(x)))], test_result[, length(x)+1])
  pre_prob <- rowSums(test_result[, -(sum(x)+1)])/sum(x)
  return(roc(data_test[, 'y'], pre_prob)$auc + 0)
}


# calculate Q statistics
# result of BLDE-RS-KNN
pre_validate <- function(x){
  validate_result <- cbind(validate_result[, which(x==rep(1, length(x)))], validate_result[, length(x)+1])
  pre_vali <- round(rowSums(validate_result[, -(sum(x)+1)])/sum(x))
  return(pre_vali) 
}

pre_test <- function(x){
  test_result <- cbind(test_result[, which(x==rep(1, length(x)))], test_result[, length(x)+1])
  pre_tes <- round(rowSums(test_result[, -(sum(x)+1)])/sum(x))
  return(pre_tes)
}

# Q statistics
Q_statistic <- function(real_value, predict_1, predict_2)
{
	n_11 <- sum((predict_1 == real_value) * (predict_2 == real_value)) 
	n_10 <- sum((predict_1 == real_value) * (predict_2 != real_value))
	n_01 <- sum((predict_1 != real_value) * (predict_2 == real_value))
	n_00 <- sum((predict_1 != real_value) * (predict_2 != real_value))
	Q <- (n_11 * n_00 - n_10 * n_01)/(n_11 * n_00 + n_10 * n_01)
	return(Q)
}


###############################################################################################
# we use the BLDE-RS-KNN to study the Australia credit data from UCI machine learning dataset

# model data
data <- read.table('Australia.txt', sep=' ')
data <- data.frame(data[,-15], y=data[,15])
data_train <- data[1:345, ]
data_validate <- data[346:551, ]
data_test <- data[552:690, ]

dim_set <- seq(2, 13)
k_set <- seq(1, 19, 3)
matrix_ncol <- length(dim_set)
matrix_nrow <- length(k_set)


# RS-KNN
S_validate_accuracy_matrix <- matrix(NA, matrix_nrow, matrix_ncol)
S_validate_auc_matrix <- matrix(NA, matrix_nrow, matrix_ncol)

S_test_accuracy_matrix <- matrix(NA, matrix_nrow, matrix_ncol)
S_test_type_1_matrix <- matrix(NA, matrix_nrow, matrix_ncol)
S_test_type_2_matrix <- matrix(NA, matrix_nrow, matrix_ncol)
S_test_auc_matrix <- matrix(NA, matrix_nrow, matrix_ncol)

# BLDE-RS-KNN
B_validate_accuracy_matrix <- matrix(NA, matrix_nrow, matrix_ncol)
B_validate_auc_matrix <- matrix(NA, matrix_nrow, matrix_ncol)

B_test_accuracy_matrix <- matrix(NA, matrix_nrow, matrix_ncol)
B_test_type_1_matrix <- matrix(NA, matrix_nrow, matrix_ncol)
B_test_type_2_matrix <- matrix(NA, matrix_nrow, matrix_ncol)
B_test_auc_matrix <- matrix(NA, matrix_nrow, matrix_ncol)


# compare two method
validate_Q_matrix <- matrix(NA, matrix_nrow, matrix_ncol)
test_Q_matrix <- matrix(NA, matrix_nrow, matrix_ncol)


# calculate all statistics
Start_time <- Sys.time()

for(n in 1:matrix_ncol){
  for(m in 1:matrix_nrow){
    print('m is:')
    print(m)
    print('n is:')
    print(n)
    # model parameter
    sub_dim <- dim_set[n]
    nresample <- 60
    K <- k_set[m]
    sub_col <- sub_col_index(data_train, sub_dim=sub_dim, nresample=nresample)
    
    # learn processing
    train_result <- class_pool(data_train, data_validate, data_test, sub_col=sub_col, nresample=nresample, K=K)[1:345, ]
    validate_result <- class_pool(data_train, data_validate, data_test, sub_col=sub_col, nresample=nresample, K=K)[346:551, ]
    test_result <- class_pool(data_train, data_validate, data_test, sub_col=sub_col, nresample=nresample, K=K)[552:690, ]
    
    # prediction of RS-KNN
    S_validate_accuracy_matrix[m, n] <- validate_accuracy(rep(1, 60))
    S_test_accuracy_matrix[m, n] <- test_accuracy(rep(1, 60))[1]
    S_test_type_1_matrix[m, n] <- test_accuracy(rep(1, 60))[2]
    S_test_type_2_matrix[m, n] <- test_accuracy(rep(1, 60))[3]
    S_validate_auc_matrix[m, n] <- validate_auc(rep(1, 60))
    S_test_auc_matrix[m, n] <- test_auc(rep(1, 60))
    
    # prediction of BLDE-RS-KNN
    optimize_vector <- BLDE_AG(30, 60, 200, 0.05, validate_accuracy)
    B_validate_accuracy_matrix[m, n] <- validate_accuracy(optimize_vector)
    B_test_accuracy_matrix[m, n] <- test_accuracy(optimize_vector)[1]
    B_test_type_1_matrix[m, n] <- test_accuracy(optimize_vector)[2]
    B_test_type_2_matrix[m, n] <- test_accuracy(optimize_vector)[3]
    B_validate_auc_matrix[m, n] <- validate_auc(optimize_vector)
    B_test_auc_matrix[m, n] <- test_auc(optimize_vector)
    
    # Q statistic
    RS_KNN_test <- pre_test(rep(1, 60))
    RS_KNN_validate <- pre_validate(rep(1, 60))
    BLDE_RS_KNN_test <- pre_test(optimize_vector)
    BLDE_RS_KNN_validate <- pre_validate(optimize_vector)
    validate_Q_matrix[m, n] <- Q_statistic(data_validate[, 'y'], RS_KNN_validate, BLDE_RS_KNN_validate) 
    test_Q_matrix[m, n] <- Q_statistic(data_test[, 'y'], RS_KNN_test, BLDE_RS_KNN_test) 
  }
}

End_time <- Sys.time()
print(End_time - Start_time)


# savefile
write.table(round(S_validate_accuracy_matrix, 4), file='S_validate_accuracy_matrix.csv', sep=',', row.names=F, col.names=F, quote=FALSE)
write.table(round(S_validate_auc_matrix, 4), file='S_validate_auc_matrix.csv', sep=',', row.names=F, col.names=F, quote=FALSE)
write.table(round(S_test_accuracy_matrix, 4) , file='S_test_accuracy_matrix .csv', sep=',', row.names=F, col.names=F, quote=FALSE)
write.table(round(S_test_type_1_matrix, 4), file='S_test_type_1_matrix.csv', sep=',', row.names=F, col.names=F, quote=FALSE)
write.table(round(S_test_type_2_matrix, 4), file='S_test_type_2_matrix.csv', sep=',', row.names=F, col.names=F, quote=FALSE)
write.table(round(S_test_auc_matrix, 4), file='S_test_auc_matrix.csv', sep=',', row.names=F, col.names=F, quote=FALSE)
write.table(round(B_validate_accuracy_matrix, 4), file='B_validate_accuracy_matrix.csv', sep=',', row.names=F, col.names=F, quote=FALSE)
write.table(round(B_validate_auc_matrix, 4), file='B_validate_auc_matrix.csv', sep=',', row.names=F, col.names=F, quote=FALSE)
write.table(round(B_test_accuracy_matrix, 4), file='B_test_accuracy_matrix.csv', sep=',', row.names=F, col.names=F, quote=FALSE)
write.table(round(B_test_type_1_matrix, 4), file='B_test_type_1_matrix.csv', sep=',', row.names=F, col.names=F, quote=FALSE)
write.table(round(B_test_type_2_matrix, 4), file='B_test_type_2_matrix.csv', sep=',', row.names=F, col.names=F, quote=FALSE)
write.table(round(B_test_auc_matrix, 4), file='B_test_auc_matrix.csv', sep=',', row.names=F, col.names=F, quote=FALSE)
write.table(round(validate_Q_matrix, 4), file='validate_Q_matrix.csv', sep=',', row.names=F, col.names=F, quote=FALSE)
write.table(round(test_Q_matrix, 4), file='test_Q_matrix.csv', sep=',', row.names=F, col.names=F, quote=FALSE)

