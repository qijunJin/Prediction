library(readr)

real <- read_delim("real.csv", ";", escape_double = FALSE, 
                   col_types = cols(date = col_date(format = "%Y-%m-%d")), 
                   trim_ws = TRUE)

predicted <- read_delim("predicted.csv", 
                        ";", escape_double = FALSE, col_types = cols(date = col_date(format = "%Y-%m-%d")), 
                        trim_ws = TRUE)


score <- function(real, predicted) {
  
  ## Compare that both datasets have the correct names of columns
  correct_names<- c("customer", "date", "billing")
  
  if(length(setdiff(correct_names, names(real)))>0|
     length(setdiff(correct_names, names(predicted)))>0){
    print("ERROR: You don't have the correct column names (customer, date, billing)")
    break;
  }
  
  ## Compare that both datasets have the same number of rows
  if(nrow(real)!=nrow(predicted)){
    
    print("ERROR: Your datasets don't have the same numbers of rows")
    break;
  }  
  
  ## Check if we have any missing value
  if(sum(is.na(real))>0 | sum(is.na(predicted))>0){
    
    print("ERROR: You have missing values!")
    break;
  }  
  
  ## Compare the results of both datasets
  x <- ifelse(test = real$date - predicted$date == 0, yes = ifelse(abs(real$billing - predicted$billing <= 10), 
                                                                   yes = 1 ,no =  0), no = 0) 
  
  ## Calculate and return accuracy
  return(paste("accuracy =",round(sum(x)/length(x),2)))
}

      
 score(real = real, predicted =  predicted)  
 