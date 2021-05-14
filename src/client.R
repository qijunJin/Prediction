pacman::p_load(
  "ConfigParser",
  "httr"
)

submit_predictions <- function(config_file, df){

  config <- ConfigParser$new()
  config <- config$read(config_file)

  protocol <- 'http://'
  host <- config$get(option='host', section='DEFAULT')
  token <- config$get(option='token', section='DEFAULT')

  # post predictions
  endpoint <- '/predictions'
  url <- paste(protocol, host, endpoint, sep='')
  headers <- add_headers(
    "Content-Type" = "application/json",
    "Authorization" = paste("Bearer", token),
    "Prefer" = "return=representation"
  )
  payload <- df[,c('customer', 'date', 'billing')]
  r <- POST(url, body = payload, encode = "json", headers)

  status <- http_status(r)$message
  return(status)

}
