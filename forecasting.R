# Use own FRED API key 
# fred_key =
  
library(fredr)
library(fpp2)
library(vars)
library(tseries)
library(forecast)
library(rstan)
library(prophet)
library(ggplot2)
library(MLmetrics)
library(tseries)
library(strucchange)

# Use own FRED API key 
fredr_set_key(fred_key)

sales    <- fredr(series_id="MRTSSM4453USN")
sales_ts <- ts(log(sales$value), start=1992, frequency=12) 

any(is.na(sales_ts))

# ADF Test 
adf.test(sales_ts)


autoplot(sales_ts) +
  ylab("Log(Sales)") + 
  ggtitle("Monthly Retail Sales of Beer, Wine, and Liquor Stores")
autoplot(stl(sales_ts, s.window="periodic"), main="STL Decomposition")
ggtsdisplay(sales_ts, main="Series with ACF and PACF") 

# Train/test split 
n     <- round((2/3)*length(sales_ts)) 

train <- ts(sales_ts[1:n], start=1992, frequency=12) 

test  <- ts(sales_ts[n+1:length(sales_ts)], start=2012 + (5/12), frequency=12)
test  <- na.remove(test)

# ARIMA
f_arima <- forecast(auto.arima(train), h=length(sales_ts)-n)

autoplot(train, series="Train") +
  autolayer(test, series="Test") + 
  autolayer(f_arima$mean, series="ARIMA") + 
  ggtitle("Forecasting with ARIMA") + 
  ylab("Log(Sales)") + 
  labs(colour="")

arima_recur <- c() 
for (i in n:length(sales_ts)) { 
  predict     <- forecast(auto.arima(ts(sales_ts[1:i], frequency=12),
                                     method="CSS"), h=12) 
  actual      <- sales_ts[(i+1):(i+12)] 
  arima_recur <- cbind(arima_recur, MAPE(predict$mean, actual))
}

arima_recur    <- na.omit(as.data.frame(t(arima_recur)))
h              <- c(1:nrow(arima_recur))
arima_recur_df <- cbind.data.frame(h, arima_recur)

arima_roll <- c()
for (i in 0:(length(sales_ts)-n)) { 
  predict    <- forecast(auto.arima(ts(sales_ts[(1+i):(n+i)],frequency=12), 
                                    method="CSS"), h=12) 
  actual     <- sales_ts[(n+i+1):(n+i+12)]
  arima_roll <- cbind(arima_roll, MAPE(predict$mean, actual))
}

arima_roll    <- na.omit(as.data.frame(t(arima_roll)))
h             <- c(1:nrow(arima_roll))
arima_roll_df <- cbind.data.frame(h, arima_roll) 

mean(arima_recur$V1)
mean(arima_roll$V1)

# ETS 
f_ets <- forecast(ets(train), h=length(sales_ts)-n)

autoplot(train, series="Train") +
  autolayer(test, series="Test") + 
  autolayer(f_ets$mean, series="ETS") + 
  ggtitle("Forecasting with ETS") + 
  ylab("Log(Sales)") + 
  labs(colour="")  

ets_recur <- c() 
for (i in n:length(sales_ts)) { 
  predict   <- forecast(ets(ts(sales_ts[1:i], frequency=12)), h=12) 
  actual    <- sales_ts[(i+1):(i+12)] 
  ets_recur <- cbind(ets_recur, MAPE(predict$mean, actual))
}

ets_recur    <- na.omit(as.data.frame(t(ets_recur)))
h            <- c(1:nrow(ets_recur))
ets_recur_df <- cbind.data.frame(h, ets_recur) 

ets_roll <- c()
for (i in 0:(length(sales_ts)-n)) { 
  predict  <- forecast(ets(ts(sales_ts[(1+i):(n+i)], frequency=12)), h=12) 
  actual   <- sales_ts[(n+i+1):(n+i+12)]
  ets_roll <- cbind(ets_roll, MAPE(predict$mean, actual))
}

ets_roll    <- na.omit(as.data.frame(t(ets_roll)))
h           <- c(1:nrow(ets_roll))
ets_roll_df <- cbind.data.frame(h, ets_roll) 

mean(ets_recur$V1)
mean(ets_roll$V1)

# Holt-Winters 

f_hw <- forecast(HoltWinters(train), h=length(sales_ts)-n)
autoplot(train, series="Train") + 
  autolayer(test, series="Test") + 
  autolayer(f_hw$mean, series="HW") + 
  labs(colour="") + 
  ggtitle("Forecasting with Holt Winters") +
  ylab("Log(Sales)")

hw_recur <- c() 
for (i in n:length(sales_ts)) { 
  predict  <- forecast(hw(ts(sales_ts[1:i], frequency=12)), h=12) 
  actual   <- sales_ts[(i+1):(i+12)] 
  hw_recur <- cbind(hw_recur, MAPE(predict$mean, actual))
}

hw_recur    <- na.omit(as.data.frame(t(hw_recur)))
h           <- c(1:nrow(hw_recur))
hw_recur_df <- cbind.data.frame(h, hw_recur) 

hw_roll <- c()
for (i in 0:(length(sales_ts)-n)) { 
  predict <- forecast(hw(ts(sales_ts[(1+i):(n+i)], frequency=12)), h=12) 
  actual  <- sales_ts[(n+i+1):(n+i+12)]
  hw_roll <- cbind(hw_roll, MAPE(predict$mean, actual))
}

hw_roll    <- na.omit(as.data.frame(t(hw_roll)))
h          <- c(1:nrow(hw_roll))
hw_roll_df <- cbind.data.frame(h, hw_roll) 

mean(hw_recur$V1)
mean(hw_roll$V1)

## Facebook Prophet 
sales_df           <- cbind.data.frame(sales$date, log(sales$value))
colnames(sales_df) <- c("ds", "y")

train_prophet <- sales_df[1:n,]
test_prophet  <- sales_df[n+1:nrow(sales_df),]

prophet      <- prophet(train_prophet)
future       <- make_future_dataframe(prophet, periods=nrow(sales_df)-n, 
                                      freq='month')
f_prophet    <- predict(prophet, future)
f_prophet_ts <- na.remove(ts(f_prophet$yhat[n+1:nrow(f_prophet)], 
                             start=2011.917, frequency=12))

autoplot(train, series="Train") + 
  autolayer(test, series="Test") + 
  autolayer(f_prophet_ts, series="FBP") + 
  ggtitle("Forecasts from Facebook Prophet") + 
  ylab("Log(Sales)") + 
  labs(colour="")

fbp_recur <- c() 
for (i in n:length(sales_ts)) { 
  df        <- sales_df[1:i,] 
  model     <- prophet(df)
  future    <- make_future_dataframe(model, periods=12, freq='month')
  f_prophet <- predict(model, future)
  predict   <- f_prophet$yhat[(nrow(f_prophet)-11):nrow(f_prophet)]
  actual    <- sales_ts[(i+1):(i+12)] 
  fbp_recur <- cbind(fbp_recur, MAPE(predict, actual))
}

fbp_recur    <- na.omit(as.data.frame(t(fbp_recur)))
h            <- c(1:nrow(fbp_recur))
fbp_recur_df <- cbind.data.frame(h, fbp_recur) 

fbp_roll <- c()
for (i in 0:(length(sales_ts)-n)) { # i in 0:119
  df        <- sales_df[(1+i):(n+i),] # train: 1 + i : 239 + i
  model     <- prophet(df)
  future    <- make_future_dataframe(model, periods=12, freq='month')
  f_prophet <- predict(model, future)
  predict   <- f_prophet$yhat[(nrow(f_prophet)-11):nrow(f_prophet)]
  actual    <- sales_ts[(n+i+1):(n+i+12)]
  fbp_roll  <- cbind(fbp_roll, MAPE(predict, actual))
}

fbp_roll    <- na.omit(as.data.frame(t(fbp_roll)))
h           <- c(1:nrow(fbp_roll))
fbp_roll_df <- cbind.data.frame(h, fbp_roll) 

mean(fbp_recur$V1)
mean(fbp_roll$V1)

# Combined Forecast 

prophet   <- prophet(train_prophet)
future    <- make_future_dataframe(prophet, periods=nrow(sales_df)-n, 
                                   freq='month')
f_prophet <- predict(prophet, future)
fbp       <- ts(na.remove(f_prophet$yhat[n+1:nrow(f_prophet)]), start=2012,
                frequency=12)

arima  <- forecast(auto.arima(train), h=length(sales_ts)-n)
ets    <- forecast(ets(train), h=length(sales_ts)-n)
holtw  <- forecast(HoltWinters(train), h=length(sales_ts)-n)
comb   <- (fbp + arima[["mean"]] + ets[["mean"]] + holtw[["mean"]])/4


autoplot(sales_ts) + 
  autolayer(arima$mean, series="ARIMA") + 
  autolayer(ets$mean, series="ETS") + 
  autolayer(holtw$mean, series="HW") +
  autolayer(fbp, series="FBP") +
  autolayer(comb, series="Combined") +
  ylab("Log(Sales)") + 
  labs(color="Forecasts") +
  ggtitle("Forecasting Using Combined Model") + 
  xlim(2008,2022) + 
  ylim(7.75, 9)

comb_recur <- c() 
for (i in n:length(sales_ts)) {
  df1 <- ts(sales_ts[1:i], frequency=12)
  f1  <- forecast(auto.arima(df1, method="CSS"), h=12)
  f2  <- forecast(ets(df1), h=12)
  f3  <- forecast(hw(df1), h=12)
  
  df2       <- sales_df[1:i,]
  m4        <- prophet(df2)
  future    <- make_future_dataframe(m4, periods=12, freq='month')
  f_prophet <- predict(m4, future)
  f4        <- f_prophet$yhat[(nrow(f_prophet)-11):nrow(f_prophet)]
  
  f1 <- as.vector(f1$mean)
  f2 <- as.vector(f2$mean)
  f3 <- as.vector(f3$mean)
  f4 <- as.vector(f4)
  
  predict <- (f1 + f2 + f3 + f4)/4
  actual  <- sales_ts[(i+1):(i+12)] 
  
  comb_recur <- cbind(comb_recur, MAPE(predict, actual))
}

comb_recur    <- na.omit(as.data.frame(t(comb_recur)))
h             <- c(1:nrow(comb_recur))
comb_recur_df <- cbind.data.frame(h, comb_recur) 

comb_roll <- c()
for (i in 0:(length(sales_ts)-n)) { 
  df1     <- ts(sales_ts[(1+i):(n+i)], frequency=12)
  f1      <- forecast(auto.arima(df1, method="CSS"), h=12)
  f2      <- forecast(ets(df1), h=12)
  f3      <- forecast(hw(df1), h=12)
  
  df2       <- sales_df[(1+i):(n+i),]
  m4        <- prophet(df2)
  future    <- make_future_dataframe(m4, periods=12, freq='month')
  f_prophet <- predict(m4, future)
  f4        <- f_prophet$yhat[(nrow(f_prophet)-11):nrow(f_prophet)]
  
  f1 <- as.vector(f1$mean)
  f2 <- as.vector(f2$mean)
  f3 <- as.vector(f3$mean)
  f4 <- as.vector(f4)
  
  predict   <- (f1 + f2 + f3 + f4)/4
  actual    <- sales_ts[(n+i+1):(n+i+12)] 
  comb_roll <- cbind(comb_roll, MAPE(predict, actual))
  
}

comb_roll    <- na.omit(as.data.frame(t(comb_roll)))
h            <- c(1:nrow(comb_roll))
comb_roll_df <- cbind.data.frame(h, comb_roll)

mean(comb_recur$V1)
mean(comb_roll$V1)

# Results of Back-testing

mape1 <- cbind.data.frame(h, arima_recur$V1, ets_recur$V1, 
                          hw_recur$V1, fbp_recur$V1, comb_recur$V1)
colnames(mape1) <- c("Iteration", "ARIMA", "ETS", "HW", "FBP", "Combined")

ggplot() + 
  geom_line(mape1, mapping=aes(x=Iteration, y=ARIMA, color="ARIMA")) + 
  geom_line(mape1, mapping=aes(x=h, y=ETS, color="ETS")) + 
  geom_line(mape1, mapping=aes(x=h, y=HW, color="HW")) + 
  geom_line(mape1, mapping=aes(x=h, y=FBP, color="FBP")) + 
  geom_line(mape1, mapping=aes(x=h, y=Combined, color="Combined")) +
  ggtitle("Recursive Backtesting, 12 Steps Ahead") + 
  ylab("MAPE") + 
  labs(colour="Model")

mape2 <- cbind.data.frame(h, arima_roll$V1, ets_roll$V1, 
                          hw_roll$V1, fbp_roll$V1, comb_roll$V1)
colnames(mape2) <- c("Iteration", "ARIMA", "ETS", "HW", "FBP", "Combined")

ggplot() + 
  geom_line(mape2, mapping=aes(x=Iteration, y=ARIMA, color="ARIMA")) + 
  geom_line(mape2, mapping=aes(x=h, y=ETS, color="ETS")) + 
  geom_line(mape2, mapping=aes(x=h, y=HW, color="HW")) + 
  geom_line(mape2, mapping=aes(x=h, y=FBP, color="FBP")) + 
  geom_line(mape2, mapping=aes(x=h, y=Combined, color="Combined")) +
  ggtitle("Rolling Window Backtesting, 12 Steps Ahead") + 
  ylab("MAPE") + 
  labs(colour="Model")

model  <- prophet(sales_df)
future <- make_future_dataframe(model, periods=12, 
                                freq='month')

f_prophet <- predict(model, future)

fbp <- ts(na.remove(f_prophet$yhat[1:(nrow(f_prophet)-12)]), 
          start=1992, frequency=12)
ari <- auto.arima(sales_ts)
ets <- ets(sales_ts)
how <- hw(sales_ts)
com <- (ari[["fitted"]] + ets[["fitted"]] + how[["fitted"]] + fbp) / 4

res <- sales_ts - com
ggtsdisplay(res, main="Residuals from Combined Model")

Box.test(res, type="Box-Pierce")
Box.test(res, type="Ljung-Box")

cusum   <- efp(res ~ 1, type="Rec-CUSUM")
bound_u <- boundary(cusum, alpha=0.05)
bound_l <- -1*boundary(cusum, alpha=0.05)
horizon <- ts(rep(0, times=length(cusum$process)), start=1992, frequency=12)

autoplot(cusum$process) + 
  autolayer(bound_u, color="red") + 
  autolayer(bound_l, color="red") + 
  autolayer(horizon, color="black") +
  ggtitle("Recursive CUSUM Plot") + 
  ylab("Emperical Fluctuation Process") 

rr <- recresid(res ~ 1)
x  <- c(1:length(rr))
ggplot() + 
  geom_line(cbind.data.frame(x, rr), mapping=aes(x=x, y=rr)) +
  ylab("Recursive Residuals") + 
  xlab("")

# Vector Auto-regression (VAR)

sales    <- fredr(series_id="MRTSSM4453USN")
sales_ts <- ts(log(sales$value), start=1992, end=2021, frequency=12) 

any(is.na(sales_ts))

grain    <- fredr(series_id="WPU012")
grain_ts <- ts(log(grain$value), start=1992, end=2021, frequency=12) 

any(is.na(grain_ts))

adf.test(sales_ts)
adf.test(grain_ts)

mod = lm(sales_ts ~ grain_ts)
res = mod$residuals
adf.test(res)

sales_ts <- diff(BoxCox(sales_ts, lambda="auto"), differences=1, na.rm=TRUE)
grain_ts <- diff(BoxCox(grain_ts, lambda="auto"), differences=1, na.rm=TRUE)

autoplot(stl(sales_ts, s.window="periodic"), main="STL Decomposition Sales")
autoplot(stl(grain_ts, s.window="periodic"), main="STL Decomposition Grain")

ggtsdisplay(sales_ts, main="Sales")
ggtsdisplay(grain_ts, main="Grain Producer Price Index")

# Train/test split 
n <- 336

train_s <- ts(sales_ts[1:n], start=1992, frequency=12) 
train_g <- ts(grain_ts[1:n], start=1992, frequency=12) 

test_s <- ts(sales_ts[n+1:length(sales_ts)], start=2020, frequency=12)
test_g <- ts(grain_ts[n+1:length(sales_ts)], start=2020, frequency=12)
test_s <- na.remove(test_s)
test_g <- na.remove(test_g)

# Model selection 
data <- data.frame(cbind(train_s, train_g)) 
VARselect(data, lag.max = 10)

var_mod <- VAR(data, p=4)
summary(var_mod)

# Granger causality test 
grangertest(train_s ~ train_g, order=4) 
grangertest(train_g ~ train_s, order=4)

# Impulse response functions 
len  <- c(1:25)

irf1 <- irf(var_mod, impulse="train_g", response="train_s", 
            n.ahead=24, ortho=TRUE, runs=1000)
irf2 <- irf(var_mod, impulse="train_s", response="train_g", 
            n.ahead=24, ortho=TRUE, runs=1000)

irf_data <- cbind.data.frame(len, irf1$irf, irf1$Lower, irf1$Upper,
                             irf2$irf, irf2$Lower, irf2$Upper)

colnames(irf_data) <- c("len", "irf1", "lower1", "upper1", 
                        "irf2", "lower2", "upper2")

ggplot() + 
  geom_line(irf_data, mapping=aes(x=len, y=irf1)) + 
  geom_line(irf_data, mapping=aes(x=len, y=lower1), color="red", linetype="dashed") + 
  geom_line(irf_data, mapping=aes(x=len, y=upper1), color="red", linetype="dashed") + 
  ggtitle("Orthogonal Impulse Response from Grain PPI") + 
  ylab("Sales Growth") + 
  xlab("Runs")

ggplot() + 
  geom_line(irf_data, mapping=aes(x=len, y=irf2)) + 
  geom_line(irf_data, mapping=aes(x=len, y=lower2), color="red", linetype="dashed") + 
  geom_line(irf_data, mapping=aes(x=len, y=upper2), color="red", linetype="dashed") + 
  ggtitle("Orthogonal Impulse Response from Sales") + 
  ylab("Grain PPI Growth") + 
  xlab("Runs")

var_forecast  <- predict(object=var_mod, n.ahead=12)
sales_forecast <- data.frame(var_forecast$fcst$train_s)

sales_f <- ts(sales_forecast$fcst, start=2020, frequency=12)
sales_l <- ts(sales_forecast$lower, start=2020, frequency=12)
sales_u <- ts(sales_forecast$upper, start=2020, frequency=12)

clrs <- c("darkgreen", "black", "red")

autoplot(sales_f, series="VAR") +
  autolayer(train_s, series="Train") +  
  autolayer(sales_l, color="lightblue") +  
  autolayer(sales_u, color="lightblue") +
  autolayer(test_s, series="Test") + 
  ylab("1st Difference") +
  ggtitle("Forecasting Sales with VAR") +
  guides(colour=guide_legend(title="")) +
  scale_color_manual(values=clrs) + 
  geom_ribbon(aes(ymax=sales_u, ymin=sales_l), fill="lightblue", alpha=.5)

mean(abs((test_s - sales_f)/(test_s)))

# Backtesting 
var_recur <- c() 
for (i in n:length(sales_ts)) { 
  data        <- data.frame(cbind(sales_ts[1:i], grain_ts[1:i])) 
  model       <- predict(object=VAR(data, p=4), n.ahead=1)
  predict     <- data.frame(model$fcst$X2)
  actual      <- sales_ts[i+1] 
  var_recur  <- cbind(var_recur, MAPE(predict$fcst, actual))
}

var_recur    <- na.omit(as.data.frame(t(var_recur)))
h             <- c(1:nrow(var_recur))
var_recur_df <- cbind.data.frame(h, var_recur)

var_roll <- c()
for (i in 0:(length(sales_ts)-n)) { 
  data      <- data.frame(cbind(sales_ts[(1+i):(n+i)], grain_ts[(1+i):(n+i)])) 
  model     <- predict(object=VAR(data, p=4), n.ahead=1)
  predict   <- data.frame(model$fcst$X2)
  actual    <- sales_ts[n+i+1]
  var_roll <- cbind(var_roll, MAPE(predict$fcst, actual))
}

var_roll    <- na.omit(as.data.frame(t(var_roll)))
h            <- c(1:nrow(var_roll))
var_roll_df <- cbind.data.frame(h, var_roll) 

mape1 <- cbind.data.frame(h, var_recur$V1)
colnames(mape1) <- c("Iteration", "VAR")

ggplot() + 
  geom_line(mape1, mapping=aes(x=Iteration, y=VAR, color="VAR")) + 
  ggtitle("Recursive Backtesting (Sales), 1 Step Ahead") + 
  ylab("MAPE") + 
  labs(colour="Model")

mean(mape1$VAR)

mape2 <- cbind.data.frame(h, var_roll$V1)
colnames(mape2) <- c("Iteration", "VAR")

ggplot() + 
  geom_line(mape2, mapping=aes(x=Iteration, y=VAR, color="VAR")) + 
  ggtitle("Rolling Window Backtesting (Sales), 1 Step Ahead") + 
  ylab("MAPE") + 
  labs(colour="Model")

mean(mape2$VAR)

# VAR Diagnostics 

var_res = var_mod$varresult$train_s$residuals
ggtsdisplay(var_res, main="Residuals from VAR Model (Furniture Production)")

Box.test(var_res, type="Box-Pierce")
Box.test(var_res, type="Ljung-Box")

var_res <- ts(var_res, start=1992, frequency=12)
cusum   <- efp(var_res ~ 1, type="Rec-CUSUM")
bound_u <- boundary(cusum, alpha=0.05)
bound_l <- -1*boundary(cusum, alpha=0.05)
horizon <- ts(rep(0, times=length(cusum$process)), start=1992, frequency=12)

autoplot(cusum$process) + 
  autolayer(bound_u, color="red") + 
  autolayer(bound_l, color="red") + 
  autolayer(horizon, color="black") +
  ggtitle("Recursive CUSUM Plot") + 
  ylab("Emperical Fluctuation Process") 