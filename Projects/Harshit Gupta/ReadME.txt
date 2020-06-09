Desciption of data:-

We have following data for predictions:-
1) Passenger ID:- 
	ID number given to passenger's data.
2) Survived (Only for train_data):- 
	Describes weather the passenger survived or not.
3) Name:- 
	Name of the passenger registered.
4) Pclass:-
	Passenger's class on the ship.
5) SibSp:- 
	Number of Siblings/Spouses onboard with passenger.
6) Parch:- 
	Number of Parents/Childerens onboard with passenger.
7) Ticket:- 
	Ticket Number provided to passenger.
8) Cabin:- 
	The Seat Number provided to passenger.
9) Fare:- 
	Fare of the voyage paid by passenger.
10) Age:- 
	Age of the passenger.
11) Sex:- 
	Gender of the passenger.
12) Embarked:-
	Country from where the passengers were embarked.

Here We have total 11 features for prediction but using some Exploratory Data Analysis some conclusions which are:-
1) From all these Features we requrire following features for better prediction:-
	1) Name:-
		From name, the Title of the passenger is useful beacause analysis shows that passengers with some specific titles have more chances of survival and some titles have very less chances of survival.
	2) Pclass:-
		Here analysis shows that Higher the class of the person, more is the chances for survival.
	3) Sex:-
		This feature is one of the most important feature beacuse analysis shows that during this situation, females are given more priority.
	4) Embarked:-
		This feature is also showing some corelations with survival in analysis.
2) The model which gives more accuracy is Random Forest Classifier.

Flow of the model working:-
1) Data Preprocessing:-
	This includes the taking care of missing data, creating dummy variables for features with classes.
2) Exploratory Data Analysis:-
	This contains Visualisation of data, creation of corelation matrix, Feature Engineering.
3) Model Selection:-
	This Section includes testing of different models on dataset.
4) Creating Final Predictions:-
	This is the final part of this flow which is to create output for submissions.

Scores:-
Final Accuracy:- 78.947 % on 50 % test set