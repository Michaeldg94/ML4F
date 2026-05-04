#####################################################
##Argimiro Arratia @2025 Computational Finance
##R HW 1  helper script Prob 2 and 3
# retrive from yahoo assets and merge them properly
# handling Goyal-Welch SP500 and fundamental indicators data
#######################################################

library("xts"); library("quantmod");library("vars");library("lubridate")
##library("lmtest") comes with vars
#Prob 2 Global asset causal network and volatility spillover
##retrieve assets from yahoo
assets=c("SPY", "XLK", "XLF", "XLE", "GLD", "SLV", "GSG", "BTC-USD", "ETH-USD", "TLT")
getSymbols(assets,src='yahoo',from="2018-01-01",to="2026-01-01")

BTCUSD = `BTC-USD`
ETHUSD = `ETH-USD`
df=merge(SPY$SPY.Adjusted, XLK$XLK.Adjusted, XLF$XLF.Adjusted, XLE$XLE.Adjusted, GLD$GLD.Adjusted, 
         SLV$SLV.Adjusted, GSG$GSG.Adjusted, TLT$TLT.Adjusted,
         BTCUSD$`BTC-USD.Adjusted`, ETHUSD$`ETH-USD.Adjusted`)

View(df)
##Check for Missing Values!!

#Prob 3
# Import variables, transform predicted variable
# Goyal-Welch (2022) with date as int yyyymm. Need lubridate to convert to correct Date
#1. Upload data as plain csv
GYm<-read.csv('goyal_welch2022Monthly.csv',sep=',',header=TRUE)
str(GYm)
##Index is char must change to numeric
GYm$Index = as.numeric(GYm$Index)
#2. convert yyyymm to Date format
GYm$yyyymm <-ym(GYm$yyyymm) ##ym() {lubridate}
##note GYm$yyyymm is char and to convert to xts cannot be part of data or else make all char (as xts can only be of one type)
GYm <- as.xts(GYm[-1],order.by = GYm$yyyymm)
head(GYm)

mt=GYm['1927/2021']
##compute log equity premium (GSPCep), 
## log returns of SP500 (logret)
logret =diff(log(mt$Index))
IndexDiv = mt$Index + mt$D12
#logretdiv <- log(IndexDiv) - log(mt$Index)
logretdiv =diff(log(IndexDiv))
logRfree = log(mt$Rfree + 1)
GSPCep <- logretdiv - logRfree
names(GSPCep) = "GSPCep"; 
#Target + Features: lags 1,2,3 of return and sqr rets
#Z = merge(GSPCep,na.trim(lag.xts(GSPCep,1)),na.trim(lag.xts(GSPCep,2)),na.trim(lag.xts(GSPCep,3)),
#             na.trim(lag.xts(GSPCep^2,1)),na.trim(lag.xts(GSPCep^2,2)),
             #add other features here,
 #            all=FALSE)

#write.csv2(Z,file="GSPCep_data.csv")

##Predictor (wanna-be) variables
# dividend-price ratio (dp)
dp <- log(mt$D12) - log(mt$Index)
# dividend-payout ratio (de)
de <- log(mt$D12) - log(mt$E12)
# earnings to price
ep <- log(mt$E12) - log(mt$Index)
## dividend yield (remove 1st since lag puts NA in 1st entry)
dy <- log(mt$D12[-1]) - log(lag.xts(mt$Index,1)[-1])
# Default yield spread (dfy)= BAA-AAA rated corporate bond yields:
dfy <- mt$BAA -mt$AAA

## from the table consider stock variance (svar), Book-to-Market (b.m)
## net equity expansion (ntis, start 1926), inflation (infl)
# Treasury Bill rates (tbl, 1920)
svar = mt$svar
bm <-mt$b.m
ntis <- mt$ntis
infl <-mt$infl
tbl <- mt$tbl

names(GSPCep) = "GSPCep"; 
names(ep) = "ep"; names(bm) = "bm"   
names(dp) = "dp"; names(svar) = "svar"  
names(dy) = "dy"; names(de) = "de"   
names(ntis) = "ntis"; names(infl) = "infl"  
names(tbl) = "tbl"; names(dfy) ="dfy"  
