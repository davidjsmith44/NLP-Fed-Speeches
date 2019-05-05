# Economic Impact of Federal Reserve Speeches


The Federal Reserve is responsible for setting short term interest rates and controlling the money supply in an attempt to keep inflation and unemployment within an acceptable range. Speeches and press releases by Federal Reserve Board members may provide insights into the Fed's future actions and have an impact on current interest rates.

Before the financial crisis of 2009, the Federal Reserve was very secretive about monetary policy. The minutes of Federal Open Market Committee (FOMC) meetings, the committee responsible for setting short term interest rates and changing the supply of money in the economy, were not released until weeks after the meetings took place. News organizations would speculate about the outcome of the FOMC meetings based on the size of Alan Greenspan’s briefcase, the then Federal Reserve Board Chairman. If the leather briefcase was packed with papers and appeared full, the media concluded the FOMC had made a change to economic policy. If the briefcase was thin, there would be no changes to the money supply.

The language of the speeches made by FOMC members was also very vague and hard to decipher. Chairman Greenspan’s warned that equity markets were exhibiting ‘irrational exuberance’, rather than simply state that they were overvalued. Careers were made deciphering ‘Fed Speek’ language into expectations of policy outcomes.

As a result of the financial crisis and under the leadership of Chairman Ben Bernanke and then Chairman Janet Yellen, the Federal Reserve began to be more transparent with the actions taken by the FOMC both in terms of the language used to discuss the economy and the timing of information being released. FOMC committee meeting minutes are now released at the end of the day the meetings take place. 

I wanted to investigate whether Federal Reserve Board Members’ speeches and the FOMC press releases had an impact on the future direction of interest rates. More specifically, if a new press release was 'different' than a previous press release, does this cause a change in the current yeild curve. Natural Language Processing (NLP) techniques are used to convert the language used by the Fed into vectors and the Euclidian distance from the most recent language relative to recent content released by the Fed was calculated and turned into a time series.

As further motivation for this study, the chart below shows the 5-year and 10-year Treasury yield during 2013. Going into 2013 the Federal Reserve had been increasing the supply of money in the economy by purchasing bonds in the open market. This was done to keep long term interest rates low in an effort to spur business investment. The vertical line represents the date Chairman Bernanke made a comment that the Federal Reserve would ‘taper’ (slow down) the rate of bond purchases in the future. The bond markets reacted to this comment by selling bonds which caused interest rates to increase over the following months. This reaction by the bond market is referred to as the ‘taper tantrum’. What is interesting to note is that the Federal Reserve did not change their policy at the time of the speech, but the market’s expectations changed. 
![taper_tantrum](https://github.com/davidjsmith44/Capstone/blob/master/src/taper_tantrum.png)

---
To measure the impact of the Federal Reserve speeches, the following steps were taken
1.	Historical speeches and press releases were web scrapped from the Federal Reserve. 
2.	The speech text was vectorized using natural language processing (NLP) tools using python’s sklearn and nltk packages.
3.	The Euclidean distance between the vectorized text of the latest speech and the most recent speech was calculated and turned into a time series.
4.	Historical daily U.S. Treasury yields were collected from Quandl’s API, transformed into forward rates and first differenced to create a stationary time series.
5.	Autoregressive Integrated Moving Average models with Exogenous Input (ARIMAX) were fit to each forward rate series to determine if the time series of Federal Reserve speech distance could explain some of the noise of the stationary forward rates.

Historical Speeches
I used two different sources for Federal Reserve.
Historical Speeches
The Federal Reserve web site contains the text of every public speech by the Board of Governor going back to 2006. There are 483 unique dates when speeches took place and there are several dates where multiple speeches took place. The topics of these speeches varies based on the audience and could range from a commencement address to the opening remarks at an insurance conference. 

In 2014 the Federal Reserve began publishing a press release after every FOMC meeting at the close of business on that date. There are about 10 meetings held every year creating a corpus of 43 press releases. These press releases are very measured and do not vary much from one  date to another, often containing only one or two words that are different. In this corpus, there are over 200,000 words but less than 700 unique words are used. Given how carefully constructed these texts are, I was hopeful that quantifying how different these texts were would provide a metric to forecast changes in interest rates.

Speeches and FOMC press releases were scraped from the Federal Reserve web site using python and BeautifulSoup.

The texts were run through a text processing pipeline where capitalization, punctuation and stop words were removed and the term frequency inverse document frequency was calculated.  The Euclidean Distance between the most recent speech/press release from the last speech or press release was calculated and turned into a time series (zeros for dates where no public speeches were made and the distance metric for the most recent speech on dates where there was a speech)

Below shows the bag of words from the last FOMC press release

![last_fed_speech](https://github.com/davidjsmith44/Capstone/blob/master/src/last_fed_speech.png)

The first differenced forward rates used over my cross validation set (for the speech text) are shown in the chart below. The changes in forward rates appear to be stationary with a constant variance. I wanted to test whether speech text could explain some of this noise.

![forward_changes over time](https://github.com/davidjsmith44/Capstone/blob/master/data/forward_changes_over_time.png)

## Time Series Models Used
- - -
ARIMA model for each interest rate

![equation](https://latex.codecogs.com/gif.latex?%5CDelta%5ED%20y_t%20%3D%20%5CSigma_%7Bi%3D1%7D%5E%7Bp%7D%20%5Cphi_i%20%5CDelta%5ED%20y_%7Bt-1%7D%20&plus;%20%5CSigma_%7Bj%3D1%7D%5E%7Bq%7D%20%5Ctheta_j%20%5Cepsilon_%7Bt-j%7D%20&plus;%20%5Cepsilon_t)

![equation](https://latex.codecogs.com/gif.latex?%5Cepsilon_t%20%5Cthicksim%20N%280%2C%20%5Csigma%5E2%29)

	where 	p is the number of autoregressive lags
		d is the degree of differencing
		q is the number of moving average lags

ARIMAX model for each forward rate
	
![equation](https://latex.codecogs.com/gif.latex?%5CDelta%5ED%20y_t%20%3D%20%5CSigma_%7Bi%3D1%7D%5E%7Bp%7D%20%5Cphi_i%20%5CDelta%5ED%20y_%7Bt-1%7D%20&plus;%20%5CSigma_%7Bj%3D1%7D%5E%7Bq%7D%20%5Ctheta_j%20%5Cepsilon_t-j%20&plus;%20%5CSigma_%7Bm%3D1%7D%5E%7BM%7D%20%5Cbeta_m%20X_%7Bm%2Ct%7D%20&plus;%20%5Cepsilon_t)

![equation](https://latex.codecogs.com/gif.latex?%5Cepsilon_t%20%5Cthicksim%20N%280%2C%20%5Csigma%5E2%29)

	where 	p is the number of autoregressive lags
		d is the degree of differencing
		q is the number of moving average lags
		X is the distance metric for the new speech/minutes release

ARIMA model on principal components of all interest rates

ARIMAX model on principal components of all interest rates

## MODEL RESULTS
---
![forward_changes over time](https://github.com/davidjsmith44/Capstone/blob/master/src/inital_pred_plot.png)

While the ARIMAX models created non-zero foreacasts for the changes in interest rates, the magnitude of these forecasts was limited to around +/- 1 basis point, which is considerably smaller than the standard deviation of the changes in interest rates. Further, the coefficients for the Federal Reserve Speech distances were all statistically insignificant.

The ARIMAX model for the 3 year forward rate demonstrated the largest forecasts, which makes sense since the Fed is slow to make policy changes that the market does not expect over in a period of slow growth.

## CONCLUSIONS/FURTHER STUDY
--- 
My initial NLP vectorization considered each word as a unique token. I had hoped that this would allow the model to pick up new words in press releases and the determine explanatory power of these words. If an FOMC speech contains the word ‘moderate’ it could mean very different things for the implications of monetary policy. For example, ‘moderate growth’ would imply that the Fed is not seeing a reason to change policy because growth is not excessive and will not have an impact on inflation. “Moderate purchases” however would signal that the Fed is planning on reducing the rate of growth of the money supply. I then went back and recreated my NLP pipeline to incorporate n-grams of up to three words as tokens to better identify the difference between groups of words. This did not change the results. Considering phrases longer than 3 words or possibly sentences may allow for the model to pick up larger differences in the meaining of the press releases.

I are measuring differences in what the Fed is saying, but not differences between what the Fed says and what the market expects after seeing the same data. Further research into other latent variables that could possibly explain the market’s expecations before a FOMC meeting to determine how different the press release is versus these expectations.
•	Include a metric for inflation expectations that can be observed every day
•	Include metrics for interest rate expectations that can be observed daily
•	Do differences in Fed speeches impact the volatility of interest rates

