# Economic Impact of Federal Reserve Speeches


The Federal Reserve is responsible for setting short term interest rates and controlling the money supply in an attempt to keep inflation and unemployment within an acceptable range. Speeches and press releases by Federal Reserve Board members may provide insights into the Fed's future actions and have an impact on current interest rates.


The Federal Reserve is responsible for setting short term interest rates and controlling the money supply in an attempt to keep inflation and unemployment within an acceptable range. Speeches and press releases by Federal Reserve Board members may provide insights into the Fed's future actions and may have an impact on changes in current interest rates.

Before the financial crisis of 2009, the Federal Reserve was very secretive about monetary policy. The minutes of Federal Open Market Committee (FOMC) meetings, the committee responsible for setting short term interest rates and changing the supply of money in the economy, were not released until weeks after the meetings took place. News organizations would speculate about the outcome of the FOMC meetings based on the size of Alan Greenspan’s briefcase, the then Federal Reserve Board Chairman. If the leather briefcase was packed with papers and appeared full, the media concluded the FOMC had made a change to economic policy. If the briefcase was thin, there would be no changes to the money supply.

The language of the speeches made by FOMC members was also very vague and hard to decipher. Chairman Greenspan’s warned that equity markets were exhibiting ‘irrational exuberance’, rather than simply state that they were overvalued. Careers were made deciphering ‘Fed Speek’ language into expectations of policy outcomes.

As a result of the financial crisis and under the leadership of Chairman Ben Bernanke and then Chairman Janet Yellen, the Federal Reserve began to be more transparent with the actions taken by the FOMC both in terms of the language used to discuss the economy and the timing of information being released. FOMC committee meeting minutes are now released at the end of the day the meetings take place. 

I wanted to investigate whether Federal Reserve Board Members’ speeches and the FOMC press releases had an impact on the future direction of interest rates.

As further motivation for this study, the chart below shows the 5-year and 10-year Treasury yield during 2013. Going into 2013 the Federal Reserve had been increasing the supply of money in the economy by purchasing bonds in the open market. This was done to keep long term interest rates low in an effort to spur business investment. The vertical line represents the date Chairman Bernanke made a comment that the Federal Reserve would ‘taper’ (slow down) the rate of bond purchases in the future. The bond markets reacted to this comment by selling bonds causing interest rates to increase over the following months. This reaction by the bond market is referred to as the ‘taper tantrum’. What is interesting to note is that the Federal Reserve did not change their policy at the time of the speech, but the market’s expectations changed. 
![taper_tantrum](https://github.com/davidjsmith44/Capstone/blob/master/src/taper_tantrum.png)

To measure the impact of the Federal Reserve speeches, the following steps were taken
1.	Historical speeches and press releases were web scrapped from the Federal Reserve. 
2.	The speech text was vectorized using natural language processing (NLP) tools using python’s sklearn and nltk packages.
3.	The Euclidean distance between the vectorized text of the latest speech and the most recent speech was calculated and turned into a time series.
4.	Historical daily U.S. Treasury yields were collected from Quandl’s API, transformed into forward rates and first differenced to create a stationary time series.
5.	Autoregressive moving average models were fit to each forward rate series to determine if the time series of Federal Reserve speech distance could explain some of the noise of the stationary forward rates.

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

MODEL RESULTS
	-no statistical significance for the coefficients on fed speeches or the FOMC press releases.
	-very little forecasted impact due to a new speech
	-largest impact is at the 3 year forward rate, which makes sense since the Fed is slow to make policy changes that the market does not expect over in a period of slow growth.

Initial predictions
![forward_changes over time](https://github.com/davidjsmith44/Capstone/blob/master/src/inital_pred_plot.png)

CONCLUSIONS/FURTHER STUDY
•	My NLP vectorization considered each word as a unique token. I had hoped that this would allow the model to pick up new words in press releases and the determine explanatory power of these words. This approach may not be appropriate to pick up the nuances of what a new word means. If an FOMC speech contains the word ‘moderate’ it could mean very different things for the implications of monetary policy. For example, ‘moderate growth’ would imply that the Fed is not seeing a reason to change policy because growth is not excessive and will not have an impact on inflation. “Moderate purchases” however would signal that the Fed is planning on reducing the rate of growth of the money supply. I intend to change my NLP pipeline to incorporate n-grams of words as tokens to better identify the difference between groups of words.

•	I are measuring differences in what the Fed is saying, but not differences between what the Fed says and what the market expects after seeing the same data. Further research into other latent variables that could possibly explain the market’s expecations before a FOMC meeting to determine how different the press release is versus these expectations.
•	Potential paths for future study
•	Include a metric for inflation expectations that can be observed every day
•	Include metrics for interest rate expectations that can be observed daily
•	Do differences in Fed speeches impact the volatility of interest rates


May be a class imbalance issue. There are few FOMC press releases (43 used here) and not every FOMC meeting results in minutes that change expectations. During the time period studied for FOMC press releases, the Federal Reserve maintained a consistent policy of reducing bond prices and 
We may not have captured a time period when the actions of the Fed had a meaningful impact on expectations.
