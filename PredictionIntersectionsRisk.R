#################################
######## Data import ###########
#################################
library(haven)
library(factoextra) 
library(elasticnet)
library(randomForest)
library(corrr)
library(mboost)
library(glmnet)
library(dplyr)
library(Rtsne)
library(plotly)
library(mpath)
library(pscl)
library(tibble)
library(caret)
library(gbm)

# Load the data
mac_mini <- ('/Users/philippebeliveau/Desktop/Notebook_Jupyter_R/Winter_2024/Advanced_stats_learning/Projet/Project-Material/data_final.csv')
macbook <- ('/Users/philippebeliveau/Desktop/Notebook/Winter_2024/Advanced_stats_learning/Project-Material/data_final.csv')

df_raw <- read.csv(macbook, sep = ";", header=TRUE)
df_raw <- subset(df_raw, select = -c(X, X.1, street_1, street_2, rue_1, rue_2, x, y, date_, half_phase, ped_100, traffic_10000))
df_raw <- na.omit(df_raw)
df_raw <- df_raw[!duplicated(df_raw$int_no), ]

# Replace spaces with underscores in the levels of the borough variable
df_raw$borough <- gsub("[-¶]", "_", df_raw$borough)

# Intersection number as index 
int_no <- df_raw$int_no
row.names(df_raw) <- df_raw$int_no
df_raw$int_no <- NULL  # Remove the "int_no" column

#### Group level ####
observations_per_level <- table(df_raw$borough)
low_count_levels <- names(observations_per_level[observations_per_level < 50])
df_raw$grouped_borough <- ifelse(df_raw$borough %in% low_count_levels, "Other", df_raw$borough)

# Analysis
df_borough <- df_raw
accidents_per_borough <- aggregate(acc ~ borough, data = df_borough, FUN = sum)
rows_per_borough <- aggregate(acc ~ borough, data = df_borough, FUN = length)
accidents_per_borough$proportion <- accidents_per_borough$acc / rows_per_borough$acc
mean(accidents_per_borough$proportion)
sum(rows_per_borough$acc)
accidents_per_borough
# Factor
df_raw$grouped_borough <- factor(df_raw$grouped_borough)
levels(df_raw$grouped_borough)
df_raw <- cbind(df_raw, model.matrix(~ grouped_borough - 1, data = df_raw))
df_raw <- subset(df_raw, select = -c(borough, grouped_borough))

##############################################
######## Scatter plot of the variable ########
##############################################
par(mfrow = c(4, 4))

for (variable in names(df_raw)) {
  # Check for finite values
  if (all(is.finite(df_raw[[variable]]))) {
    plot(df_raw[[variable]], (df_raw$acc), main = paste("Scatter plot of", variable, "vs Accidents"),
         xlab = variable, ylab = "Number of Accidents", col = "blue")
  } else {
    cat("Skipping variable", variable, "due to non-finite values.\n")
  }
}

############################################
########## Labeling for TSNE #############
###########################################

# Define the thresholds
not_dangerous_threshold <- 1
relatively_dangerous_threshold <- 8

# Categorize intersections based on 'x' and assign numerical values
dataset <- df_raw %>%
  mutate(safety_category = cut(acc, 
                               breaks = c(-Inf, not_dangerous_threshold, relatively_dangerous_threshold, Inf),
                               labels = c("Not Dangerous", "Relatively Dangerous", "Very Dangerous"),
                               include.lowest = TRUE),
         danger_level = match(safety_category, c("Not Dangerous", "Relatively Dangerous", "Very Dangerous")))

# Standardize numeric columns
scaled_df <- dataset
numeric_columns <- sapply(scaled_df, is.numeric)
scaled_df[numeric_columns] <- scale(scaled_df[numeric_columns])

################################
######### Scaling #############
###############################

datpca = as.matrix(df_raw)
datpca=apply(datpca,2,scale)

# Keeping int_no as index
datpca <- cbind(int_no, as.data.frame(datpca))
rownames(datpca) <- datpca$int_no
datpca <- datpca[, -which(names(datpca) == "int_no")]
datpca= na.omit(datpca)

############################################
########## TSNE Visualization #############
###########################################

exclude_columns <- c("danger_level")
numeric_columns <- setdiff(names(scaled_df), exclude_columns)
numeric_data <- scaled_df[, numeric_columns]
tsne_df <- na.omit(numeric_data)
scaled_df <- na.omit(scaled_df)

# Run t-SNE
tsne_result <- Rtsne(tsne_df, perplexity = 30, dims = 2, check_duplicates = FALSE)
tsne_data <- data.frame(tsne_result$Y, label = as.factor(scaled_df$safety_category))
plot_ly(data = tsne_data, x = ~X1, y = ~X2, color = ~label, type = "scatter", mode = "markers")

############################################
##### Principal components analysis ########
###########################################
datpca <- subset(datpca, select = -c(acc))
pca=prcomp(datpca) 
pca$x
get_eig(pca)


# Visualization
fviz_eig(pca, choice = c("variance"), ncp= 15, addlabels = TRUE)

set.seed(4689)
# Lasso-type regularization
spcafit=spca(datpca, K=35, sparse="penalty",para=rep(32, 35))
sum(spcafit$pev)

#######################
#### PCA LOADINGS #####
#######################
loadings_matrix <- spcafit$loadings
loadings_matrix
variable_names <- rownames(loadings_matrix)
top_loadings_df <- data.frame(Component = integer(), Variable = character(), Loading = numeric())

for (i in 1:ncol(loadings_matrix)) {
  # Get the loadings for the current component
  component_loadings <- loadings_matrix[, i]
  
  # Get the indices of the top three variables with the highest loadings
  top_three_indices <- order(abs(component_loadings), decreasing = TRUE)[1:3]
  
  # Extract the names and loadings of the top three variables
  top_three_variables <- variable_names[top_three_indices]
  top_three_loadings <- component_loadings[top_three_indices]
  
  # Create a dataframe for the top three variables of this component
  component_df <- data.frame(Component = rep(i, 3),
                             Variable = top_three_variables,
                             Loading = top_three_loadings)
  
  # Add the dataframe for this component to the main dataframe
  top_loadings_df <- rbind(top_loadings_df, component_df)
}
top_loadings_df$Loading <- round(top_loadings_df$Loading, 2)

rearranged_df <- data.frame(Component = integer(), Variable = character(), Loading = numeric())
for (i in unique(top_loadings_df$Component)) {
  # Extract the rows corresponding to the current component
  component_rows <- top_loadings_df[top_loadings_df$Component == i, ]
  
  # Create a new row for the rearranged dataframe
  new_row <- c(Component = i, 
               Variable = paste(component_rows$Variable, collapse = "/"), 
               Loading = paste(component_rows$Loading, collapse = "/"))
  
  # Append the new row to the rearranged dataframe
  rearranged_df <- rbind(rearranged_df, new_row)
}
pcs <- rearranged_df

######################################################
######## TRAIN TEST SPLIT (ORGINAL VARIABLES) ########
######################################################

####### For GLM Model ######
set.seed(4689)
df_raw <- na.omit(df_raw)
n <- nrow(df_raw)
df <- df_raw[sample(n), ]

train_index <- round(0.80 * n)
train_set <- df[1:train_index, ]
x_train <- model.matrix(acc ~ . - 1, data = train_set) # '- 1' to exclude the intercept
y_train <- train_set$acc

test_set <- df[(train_index + 1):n, ]
test_setX <- as.matrix(test_set)
x_test <- model.matrix(acc ~ . - 1, data = test_set) 
y_test<- test_set$acc

######### Tree base Model - ORGINAL VARIABLES SPLIT #########
set.seed(4689)
# Define the set
train_set <- df[1:train_index, ]
test_set <- df[(train_index + 1):n, ]
cat("Training set size:", nrow(train_set), "\n")
cat("Test set size:", nrow(test_set), "\n")

######### Tree base Model - Principal Component SPLIT ######
loadings <- spcafit$loadings
datpca_transformed <-  as.data.frame((as.matrix(datpca) %*% loadings))
data_pca <- cbind(as.data.frame(datpca_transformed), acc = df_raw$acc)

set.seed(4689)
n <- nrow(data_pca)
dfpca <- data_pca[sample(n), ]
train_setPCA <- dfpca[1:train_index, ]
test_setPCA <- dfpca[(train_index + 1):n, ]
##########################################################################################
######################## MODELING - Generalized linear model #############################
##########################################################################################

##################################
############ Poisson #############
##################################
options(scipen = 999)
set.seed(4689)
par(mfrow = c(1, 1))
help(glmnet)
plot(glmnet(train_set, train_set$acc, family="poisson",alpha=1,  standardize = TRUE),xvar = "lambda", label = TRUE)
cvgerlasso <- cv.glmnet(x_train, train_set$acc, family="poisson", alpha=1)
plot(cvgerlasso, xlab = "Lambda", ylab = "Deviance")

# Look at the lambda that gives the minimum mean cross-validated error
best_lambdaPoisson <- cvgerlasso$lambda.min
best_lambdaPoisson

# Fit the final model using the best lambda
final_model_Poisson <- glmnet(x_train, y_train, family = "poisson", alpha = 1, lambda = best_lambdaPoisson)

#### Coefficient #####
coeflassoger=predict(final_model_Poisson,newx=x_test, s=best_lambdaPoisson,type="coefficients")
length(coeflassoger[coeflassoger[,1] != 0,])

coef_matrix <- as.matrix(coeflassoger)
coef_df <- as.data.frame(coef_matrix)
coef_df

#### Prediction #####
set.seed(4689)
predlassoger=(predict(final_model_Poisson,newx=x_test,s = best_lambdaPoisson, type = "response"))
predictions_vec <- as.vector(predlassoger)
y_test <- test_set$acc

mae <- mean(abs(predictions_vec - y_test))
mse <- mean((predictions_vec - y_test)^2)
rmse <- sqrt(mse)
# Print the metrics
print(paste("MAE:", mae))
print(paste("MSE:", mse))
print(paste("RMSE:", rmse))

# Log-likelihood
log_likelihood <- log(dpois(y_test, lambda = predlassoger))
total_log_likelihood <- sum(log_likelihood)
print(total_log_likelihood)

#######################################
############ Negative Binomial ########
#######################################
# there was one NA in ln_dist coz dist was 0, replace it with 0
na_locations <- which(is.na(train_set), arr.ind = TRUE)

##### Fit the negative binomial model with Lasso penalty #####
set.seed(4689)
cv_fit_nb <- cv.glmregNB(acc~., data = train_set, alpha =1, plot.it = FALSE)

#### Lambdfit#### Lambda #### 
best_lambda_nb <- cv_fit_nb$lambda.optim
best_lambda_nb

# Fit the final model using the best lambda
final_model_nb <- glmregNB(acc~., data = train_set, alpha = 1, lambda = best_lambda_nb)
#Xtest <- df[(train_index + 1):n, -which(names(df) == "acc")]

##### Prediction #####
predictions_nb <- round(predict(final_model_nb, newx = test_set, s = best_lambda, type = "response")) #Xtest?

##### Coefficients ##### 
non_zero_count <- sum(final_model_nb$beta != 0)
print(non_zero_count)
final_model_nb$beta

##### Performance measure #####
predictions_rounded_vec_nb <- as.vector(predictions_nb)
mae_nb <- mean(abs(predictions_rounded_vec_nb - test_set$acc))
mse_nb <- mean((predictions_rounded_vec_nb - test_set$acc)^2)
rmse_nb <- sqrt(mse_nb)
cat("MAE:", mae_nb, "\n")
cat("MSE:", mse_nb, "\n")
cat("RMSE:", rmse_nb, "\n")

# Log-likelihood
log_likelihood_NB<- log(dpois(test_set$acc, lambda = predictions_nb))
total_log_likelihood_NB <- sum(log_likelihood_NB)
print(total_log_likelihood_NB)

###########################################
############ Zero inflated poisson ########
###########################################
set.seed(4689)
beta_matrix <- as.matrix(final_model_Poisson$beta)

# Convert the matrix to a data frame
beta_df <- as.data.frame(beta_matrix)
beta_df <- tibble::rownames_to_column(beta_df, var = "Coefficient")
names(beta_df) <- c("Coefficient", "Value")

selected_coefficients <- beta_df[beta_df$Value != 0,]
selected_coefficient_names <- selected_coefficients$Coefficient
train_set_filtered <- train_set[, colnames(train_set) %in% selected_coefficient_names]
test_set_filtered <- test_set[, colnames(test_set) %in% selected_coefficient_names]

# Model 
zip_model <- zeroinfl(y_train ~ . | ., data = train_set_filtered, dist = "negbin", model=T, y=T, x=T)
length(zip_model$coefficients$zero)
zip_model
# Prediction 
pred_zeroIn <- predict(zip_model, test_set_filtered)

#### Performance Measure ####
pred_zeroIn_vector <- as.vector(pred_zeroIn)
mae_nb <- mean(abs(pred_zeroIn_vector - test_set$acc))
mse_nb <- mean((pred_zeroIn_vector - test_set$acc)^2)
rmse_nb <- sqrt(mse_nb)
cat("MAE:", mae_nb, "\n")
cat("MSE:", mse_nb, "\n")
cat("RMSE:", rmse_nb, "\n")

# Log-likelihood
log_likelihood_zeroIN <- log(dpois(test_set$acc, lambda = pred_zeroIn))
total_log_likelihood_zeroIN <- sum(log_likelihood_zeroIN)
print(total_log_likelihood_zeroIN)

##########################################################################################
############################## MODELING - Tree base model  ###############################
##########################################################################################

#####################################################
######## RANDOM FOREST - With PCA Approach ##########
#####################################################

#### TUNING #######
# # mtry
# tune_result <- tuneRF(x = train_setPCA[, -which(names(train_setPCA) == "acc")], y = train_setPCA$acc)
# print(tune_result)
# 
# # ntree
# # Define the range of ntree values
# ntree_values <- c(100, 200, 300, 400, 500, 600, 700, 800, 1000)
# train_errors <- numeric(length(ntree_values))
# 
# # Perform model fitting with different ntree values and collect training errors
# for (i in seq_along(ntree_values)) {
#   # Train the random forest model with current ntree value
#   rf <- randomForest(acc ~ ., data = train_setPCA, ntree = ntree_values[i], mtry = 11)
#   # Make predictions on the training set
#   predicted <- predict(rf)
#   # Calculate the training error (MSE)
#   train_errors[i] <- mean((predicted - train_setPCA$acc)^2)
# }
# 
# # Plot the OOB error against the number of trees
# plot(ntree_values, train_errors, type = "l", xlab = "Number of Trees", ylab = "OOB Error",
#      main = "OOB Error vs. Number of Trees")

#### Random forest fit - PCA ####
set.seed(4689)

rf_pca=randomForest(acc~.,data=train_setPCA, ntree=700, mtry=11)

# Prediction
predrf_pca=round(predict(rf_pca, newdata=test_setPCA))

# Performance
errrf_pca=data.frame(mean(abs(predrf_pca-test_setPCA$acc)),
                     mean((predrf_pca-test_setPCA$acc)^2))
names(errrf_pca)=c("MAE","MSE")
row.names(errrf_pca)=c("random forest")
errrf_pca

#### Variable importance ####
virf=importance(rf_pca)
varImpPlot(rf_pca, main = 'Random forest with PCs')
help(varImpPlot)
co=correlate(train_setPCA)
coy=as.data.frame(focus(co,acc))
coy[,2]=abs(coy[,2])
coys=coy[order(-coy[,2]),,drop=FALSE]
coys[1:10,]

####################################################################
######## RANDOM FOREST - With Original variables Approach ##########
####################################################################

#### TUNING #######
# # mtry
# tune_resultRF <- tuneRF(x = train_set[, -which(names(train_set) == "acc")], y = train_set$acc)
# print(tune_resultRF)
# 
# # ntree
# train_errors <- numeric(length(ntree_values))
# for (i in seq_along(ntree_values)) {
#   # Train the random forest model with current ntree value
#   rf <- randomForest(acc ~ ., data = train_set, ntree = ntree_values[i], mtry = 11)
#   # Make predictions on the training set
#   predicted <- predict(rf)
#   # Calculate the training error (MSE)
#   train_errors[i] <- mean((predicted - train_set$acc)^2)
# }
# 
# # Plot the training errors against the number of trees
# plot(ntree_values, train_errors, type = "l", xlab = "Number of Trees", ylab = "Training Error",
#      main = "Training Error vs. Number of Trees")

##### Random forest - Original variables ######
set.seed(4689)
rf=randomForest(acc~ . ,data=train_set, ntree=500,mtry=10)

# Prediction
predrf <- round(predict(rf, newdata = test_set))

# Performance measure 
errrf=data.frame(mean(abs(predrf-test_set$acc)),
                 mean((predrf-test_set$acc)^2))
names(errrf)=c("MAE","MSE")
row.names(errrf)=c("random forest")
errrf

# Variable importance 
virf=importance(rf)
varImpPlot(rf)

################################################################
############# Boosting Regression - With PCA Approach ##########
################################################################
set.seed(4689)

glmboostgc_PCA=glmboost(acc~.,data=train_setPCA,family=Poisson(),
                        control = boost_control(mstop = 2000))

# Cross validation to find iterations
glmboostgccv_PCA=cvrisk(glmboostgc_PCA)
plot(glmboostgccv_PCA, main = '25-fold Bootstrap for PCA')
bestm_PCA=mstop(glmboostgccv_PCA)
bestm_PCA

# Coefficients 
coef(glmboostgc_PCA[bestm_PCA])
length(coef(glmboostgc_PCA[bestm_PCA]))

# Predictions
predglmboost_PCA=round(predict(glmboostgc_PCA[bestm_PCA],new=test_setPCA))


# Performance measure
errglmboost_PCA=data.frame(mean(abs(predglmboost_PCA-test_setPCA$acc)),
                           mean((predglmboost_PCA-test_setPCA$acc)^2))
names(errglmboost_PCA)=c("MAE","MSE")
row.names(errglmboost_PCA)=c("LS Boosting with glmboost")
errglmboost_PCA

###############################################################################
############ BOOSTING Regression - With Original variables Approach ###########
###############################################################################
set.seed(4689)
glmboostgc=glmboost(acc~.,data=train_set,family=Poisson(),
                    control = boost_control(mstop = 2000))

# Cross validation to find iterations
glmboostgccv=cvrisk(glmboostgc)
plot(glmboostgccv, main = '25-fold Bootstrap for Original Variables')
bestm=mstop(glmboostgccv)
bestm

# Coefficients
length(coef(glmboostgc[bestm]))

# Predictions
predglmboost=round(predict(glmboostgc[bestm],new=test_set))

# Performance measure
errglmboost=data.frame(mean(abs(predglmboost-test_set$acc)),
                       mean((predglmboost-test_set$acc)^2))
names(errglmboost)=c("MAE","MSE")
row.names(errglmboost)=c("LS Boosting with glmboost")
errglmboost

###############################################################################
#################### BOOSTING Trees - With Principal components ###############
###############################################################################

# # Tuning 
# param_grid <- expand.grid(n.trees = c(500, 600, 700, 800),
#                           interaction.depth = c(6, 7, 8),
#                           shrinkage = c(0.01, 0.05),
#                           n.minobsinnode = c(20, 30, 40))
# train_control <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation
# gbm_modelPCA <- train(acc ~ ., data = train_setPCA, method = "gbm", distribution = "poisson",
#                    trControl = train_control, tuneGrid = param_grid,
#                    verbose = FALSE)
# print(paste("Best number of trees:", gbm_modelPCA$bestTune$n.trees))
# print(paste("Best interaction depth:", gbm_modelPCA$bestTune$interaction.depth))
# print(paste("Best shrinkage:", gbm_modelPCA$bestTune$shrinkage))
# print(paste("Best n.minobsinnode:", gbm_modelPCA$bestTune$n.minobsinnode))
# print(paste("Best cross-validated RMSE:", min(gbm_modelPCA$results$RMSE)))

########## MODELING #############
set.seed(4689)
gbmgcPCA=gbm(acc~.,data=train_setPCA,distribution="poisson",
          n.trees=500,interaction.depth = 7,shrinkage =0.01, n.minobsinnode = 30)

# Get variable importance
par(mar = c(2, 2, 2, 2))
importance <- summary(gbmgcPCA)$importance
summary(gbmgcPCA)
# Predictions
predgbmPCA=predict(gbmgcPCA,newdata=test_setPCA)

# Performance measure
errgbm=data.frame(mean(abs(predgbmPCA-test_setPCA$acc)),
                  mean((predgbmPCA-test_setPCA$acc)^2))
names(errgbm)=c("MAE","MSE")
row.names(errgbm)=c("Tree boosting with gbm")
errgbm

###############################################################################
############ BOOSTING Trees - With Original variables Approach ################
###############################################################################
# Tuning 
# param_grid <- expand.grid(n.trees = c(500, 600, 700, 800),
#                           interaction.depth = c(6, 7, 8),
#                           shrinkage = c(0.01, 0.05),
#                           n.minobsinnode = c(20, 30, 40))
# train_control <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation
# gbm_model <- train(acc ~ ., data = train_set, method = "gbm", distribution = "poisson",
#                    trControl = train_control, tuneGrid = param_grid,
#                    verbose = FALSE)
# print(paste("Best number of trees:", gbm_model$bestTune$n.trees))
# print(paste("Best interaction depth:", gbm_model$bestTune$interaction.depth))
# print(paste("Best shrinkage:", gbm_model$bestTune$shrinkage))
# print(paste("Best n.minobsinnode:", gbm_model$bestTune$n.minobsinnode))
# print(paste("Best cross-validated RMSE:", min(gbm_model$results$MSE)))
# print(paste("Best cross-validated RMSE:", min(gbm_model$results$MAE)))

########## MODELING #############
set.seed(4689)
gbmgc=gbm(acc~.,data=train_set,distribution="poisson",
          n.trees=500,interaction.depth = 6,shrinkage =0.01, n.minobsinnode = 30)

# Get variable importance
importance <- summary(gbmgc)$importance
summary(gbmgc)
# Predictions
predgbm=predict(gbmgc,newdata=test_set,n.trees=500)

# Performance measure
errgbm=data.frame(mean(abs(predgbm-test_set$acc)),
                  mean((predgbm-test_set$acc)^2))
names(errgbm)=c("MAE","MSE")
row.names(errgbm)=c("Tree boosting with gbm")
errgbm

##############################################
######## Final prediction and ranking ########
##############################################

#### Random forest #####
# Prediction
set.seed(4689)
prediction_wholeSet=(predict(rf_pca, newdata=dfpca))
head(prediction_wholeSet)
# Rank the predictions in decreasing order
ranked_indices <- order(prediction_wholeSet, decreasing = TRUE)
ranked_predictions <- prediction_wholeSet[ranked_indices]
head(ranked_predictions)
ranked_df <- data.frame(
  "Intersection_number" = as.integer(names(ranked_predictions)),
  "Estimated_Accident" =  round(ranked_predictions, 2)
)
names(ranked_predictions)
write.csv(ranked_df, file = "/Users/philippebeliveau/Library/Mobile Documents/com~apple~CloudDocs/_Bureau_/Master/Winter_2024/Advanced_Stats_Learning/Projet/Rank/ranked_predictions.csv", row.names = FALSE)

#### FLAGGED INTERSECTIONS #####
# Round prediction 
roundedPred <- round(prediction_wholeSet)
# Plot residuals
residuals <- roundedPred - dfpca$acc
plot(residuals, xlab = "Observation", ylab = "Residual", main = "Residual Plot")

# Dataframe of actual vs predicted
predictions_df <- data.frame(Rounded_Predictions = roundedPred, Actual_Values = dfpca$acc)
rownames(predictions_df) <- rownames(dfpca)
positive_residuals <- predictions_df[predictions_df$s1 > 2 * predictions_df$Actual_Values, ]

##### Performance #####
final_error =data.frame(mean(abs(residuals)),
                     mean((residuals)^2))
names(final_error)=c("MAE","MSE")
row.names(final_error)=c("random forest")
final_error

##### Analysis #####

# Positive residuals analysis
residuals_df <- data.frame(Res = residuals)
positive_residuals <- residuals_df[residuals_df$s1 > 3, , drop = FALSE]
print(positive_residuals, row.names = TRUE)

### Dangerous intersections statistics ####
dangerous_intersections <- c(10, 906, 481, 842, 633, 938, 928, 6736, 931, 601, 8256, 708, 778, 444, 886, 361, 317, 190, 416, 386, 428, 863, 1092, 356, 1093, 539, 137)
dangerous_intersections_res <- residuals_df[as.character(dangerous_intersections), , drop = FALSE]
dangerous_intersections_df <- df_raw[as.character(dangerous_intersections), , drop=FALSE]
summary(dangerous_intersections_df)

### Safe intersections statistics ####
safe_intersections <- c(1665, 1216, 280, 12625, 752, 8752, 13322, 13258, 8872, 962, 619, 1680, 994, 718, 1322, 1768, 1406, 13671, 9676, 1228, 723, 1410, 10738, 1534)
safe_intersections_res <- residuals_df[as.character(safe_intersections), , drop = FALSE]
safe_intersections_df <- df_raw[as.character(safe_intersections), , drop=FALSE]
summary(safe_intersections_df)

# Calculate mean for each variable in dangerous_intersections_df
dangerous_means <- colMeans(dangerous_intersections_df, na.rm = TRUE)
safe_means <- colMeans(safe_intersections_df, na.rm = TRUE)
comparison_df <- data.frame(
  Variable = names(dangerous_means),
  Dangerous_Mean = dangerous_means,
  Safe_Mean = safe_means,
  Difference = dangerous_means - safe_means
)

# Statistic of borough
predictions_df$row_names <- rownames(predictions_df)
df_borough$row_names <- rownames(df_borough)
merged_df <- merge(predictions_df, df_borough, by.x = "row_names", by.y = "row_names", all = TRUE)
merged_df$row.names <- NULL