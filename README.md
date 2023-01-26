NBA Age Decline Project
=======================

In this project, I had the goal of researching NBA players' decline due to age. I wanted to find insights that could help predict when a player will "fall off a cliff". In this project, I performed my research using Python, Pandas, Matplotlib, Seaborn, Sci-kit learn, and the [NBA API](https://github.com/swar/nba_api) (which takes information from the NBA official stats website).

Summary
-------

This project is broken down into three parts, in which you can find the links to below for full code and project. The rest of this readme will summarize key activities and findings from each section.

-   [1\. Data Scraping](https://github.com/kevinkietle/NBA-Age-Decline-Project/blob/main/1.%20NBA%20Age%20Decline%20-%20Data%20Scraping.ipynb)

-   [2\. Data Cleaning](https://github.com/kevinkietle/NBA-Age-Decline-Project/blob/main/2.%20NBA%20Age%20Decline%20-%20Data%20Cleaning.ipynb)

-   [3\. EDA and Modeling](https://github.com/kevinkietle/NBA-Age-Decline-Project/blob/main/3.%20NBA%20Age%20Decline%20-%20EDA%20and%20Modeling.ipynb)

### Data Scraping

In this section, I explored the NBA API, and specifically the careerplayerstats and playerprofilev2 endpoints. I looped through the player database on the NBA API to find players who had at least one season on record in which they were at least 33 years old and this season had to occur no earlier than the 2012-2013 season. Then with this list of players, I retrieved yearly stats for each player using the careerplayerstats endpoint, career averages using the careerplayerstats endpoint, and yearly efficiency ranks for each player using the playerprofilev2 endpoint.

My rationale for these decisions were as follows:

-   I wanted to get players over 33 to make it highly likely that the players I am analyzing have already declined due to age. I also didn't want to get players who were out of the league young as that was likely a talent issue and not age.

-   In narrowing the scope to players over 33 at some point since 2012-13, I was able to remove any noise with regard to changing eras in basketball. This data is more likely to be of use now that we have limited the time period.

-   I will discuss actions taken with the three dataframes resulted from the endpoints, player season averages, player career averages, and player season efficiency rank in the Data Cleaning section. Efficiency is what I will be using in this project as a catch all ranking of the NBA players. There are plenty of metrics that try to do this as well such as PIE, PER, LEBRON, RAPTOR, etc. Efficiency just happens to be most readily available through the API, but other preferred metrics would likely yield similar results.

### Data Cleaning

The biggest action taken in this section was creating the EFF_CHANGE column within the dataframe for the player season ranks. If a player was the 20th ranked player in 2014 and was the 50th ranked player in 2015, a -30 would be inputted for 2014. This indicates that following 2014, a drop off of 30 spots was to occur. Assigning the drop to the previous year was an intentional decision to make this model more forward-looking and not retroactive.

In the notebook for this section, you will also see that I noted a mistake I made in the EFF_CHANGE step. I overlooked creating a groupby for the players in the dataframe, meaning the EFF_CHANGE value for the final season for all players is computed as the change from that season's rank to the next player's rank in the first season. Luckily, for most players this is not the biggest drop year to year and that is all that matters as we try to predict the year in which players "fall off a cliff".

After further cleaning, we end up with two dataframes:

-   Classification_df: This contains every selected player and an individual row for each of their seasons. Each row then contains their age, position, stats from a single season, the rank for that season, as well as the drop about to occur following that season. A column named 'Target' is also created that is assigned true if a player's drop off for a season is the highest across all their seasons. This dataframe will be used to run through classification models to see if we can predict what seasons will be deemed 'True' (meaning big drop off coming).

-   Regression_df: This contains every selected player, but each player only appears in one row. That row contains the age in the year preceding the drop off, career averages, and position. This dataframe will be used to run through regression models to see if we can predict the age in which a player drops off based on their position and career averages.

### EDA and Modeling

In this section, we created quick charts and graphs to visualize the relationship between various stats with EFF_CHANGE or rank. You can see a few that stand out in the Insights section below.

Using the two dataframes discussed in the Data Cleaning section, here are the models we ran:

-   **Multiple Linear regression**: This proved to be the most successful model, with an R-score of 0.197. In finding features with low p-values, we discovered that the following features stood out as having true impact on the age of a player's drop off: career FG%, career games played, and position. You can see this a little more in the Insights.

-   Logistic regression: This was used to try to predict what seasons would be a player's big drop off, using the classification_df dataframe. This did not work as intended as the model essentially predicted every season as False. If you recall, True means the player's biggest drop off was to precede that year. We have no use in a model that predicts False every time.

-   Random forest: Another approach to the same classification goal described in the logistic regression model. Similarly, this model always predicted False as the way to reduce error. This dataset I created for classification just ended up being too imbalanced between False/True to make any classification model work.

-   Deep learning: This was used to try to predict the age of a drop off, similar to the linear regression. Oddly enough, this model underestimated the drop off age in all instances. My understanding of deep learning models is low so the model construction was probably poor.

Insights
--------
-   Looking again, it seems the amount a player drops off is pretty inconsistent with that player's rank. However, you see that towards the right of the plot the drop off is less. Intuitively, this makes sense: if a player is already towards the bottom of the NBA player ranks, they have less room to fall further. We do also see a bit of a cluster in the top left that may suggest the best players may experience a smaller drop off, a more gradual decline.

![screenshot 1](https://user-images.githubusercontent.com/82183590/214757093-f1a9046a-d3a8-4e26-8031-6c89d05795ab.JPG)

-   The box plots below show a similar pattern. Those who score the least and the most have less negative efficiency changes (drop off). Points is a key output stat that likely has high correlation with efficiency rank, so that makes sense.

![screenshot 2](https://user-images.githubusercontent.com/82183590/214757113-f4c6015e-f3b2-4ab3-a732-220bb2e3a25b.JPG)

-   The following boxplots show that with exception to the final quartile (ages 33-38), as players get older their major drop off tends to be sharper. This could make sense as players whose most significant drop comes earlier likely declined more gradually as opposed to all at once. Remember, for this data set I only included players who had a season on record in which they were over 33 years old. As a result, some players may have had that very significant drop at younger ages but they wouldn't be included here if they did not stay in the league until age 33.

![screenshot 3](https://user-images.githubusercontent.com/82183590/214757131-b524281e-c148-4780-811f-cd4690c5b4fb.JPG)

-   Looking at all the features and their correlation to efficiency change, the most highly correlated is player age, which we explored. Games played is the next highest correlated, and this makes sense as it is highly related to player age. Interestingly, the rebounding stats at the bottom have the highest positive correlations to the drop off. As we know, big men are typically the ones getting more rebounds. Therefore, this could mean that big men see a less sharp drop (positive correlation means a smaller negative number). My guess is that as we have moved into a small ball era in the last decade, big men begin lower in ranks to begin with, meaning less room to drop

-   Look showing the highest rebounders drop the least.

![screenshot 4](https://user-images.githubusercontent.com/82183590/214757150-6c9ca2c9-34af-42eb-97f9-c42b9774cea1.JPG)

-   This look below contradicts my guess that big men start lower in the ranks. In fact, they begin as better players (looking at C and PF). My new hypothesis is that a lot of big men did not make the 33 year old threshold of my data set as the league has gone more small ball. Another chart in this section shows that center is the position with the least data points in this data set. Therefore, the bigs that did make it were very above average players. This coupled with an earlier insight that better players drop off less, make sense.

![screenshot 5](https://user-images.githubusercontent.com/82183590/214757174-4bc6557b-3003-4193-a579-77997c69663d.JPG)

-   If this model were to be made into an equation, the coefficients of each feature are listed in section 3. FGM and FG% have large coefficient values, as well as FG3M and FG3%. This means the higher a scorer they were, the younger they drop off. This could make sense as role players who score less typically can see a more gradual decline. All the positions also have a negative coefficient with center being the baseline, which means their big drop off happens younger than centers, which supports the initial hypothesis that only very good big men made it to the data set and thus lasted longer.

-   We ran p-values to see which features are significant, meaning we trust the coefficients (the effects) seen. We found that FG%, GP, and FT% are all statistically significant as well as some of the positional categories. As you can see, the PF position has low significance in terms of its difference to the center position, meaning our assumption of lumping them together in our big men hypotheses make sense.

Caveats
-------

There are a few notes I would like to include about this project:

-   Because of the mistake I mentioned in the Data Cleaning section, it is possible these results are not as conclusive as they should be. In the cases of players who are affected by the error, their drop off year was likely a year later than it should have been for the data.

-   Through the different steps of cleaning data, sometimes a player did not have their ranking data for certain seasons. As a result, their true drop off age could have been lost. Additionally, a good number of rankings for players were already NaN, meaning they may not have qualified to be ranked that year.

-   Because my method involves looking at the maximum drop off, it is possible that some players had a very bad season for some reason and that season did not end up signaling the end for them. In scanning the data, however, this seems highly unlikely for many players.

-   As mentioned, efficiency rank is the catch all metric used to evaluate effectiveness. If you don't trust the efficiency metric at in terms of ranking players relative to one another, this method should be reproduced with your metric of choice (PER, PIE, RAPTOR, LEBRON, APM, etc.)

-   An offshoot of this project that could be interesting could be to look at vitals instead of production. This means looking at player heights, weights, verticals, etc. to see when they drop off. Dependence on athleticism is often cited as a reason players decline faster.

Hope you enjoyed reading through this project!

Feel free to contact me with any questions or inquiries.

LinkedIn - <https://www.linkedin.com/in/kevinkietle/>

Email - <kevinkietle@gmail.com>

