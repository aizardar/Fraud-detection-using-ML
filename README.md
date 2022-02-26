# Machine Learning in Financial Fraud Detection

Financial fraud activities have soared despite the advancement of fraud detection models empowered by
machine learning (ML). In this project, I show how feature engineering based on time and amount of transaction can help to further improve the performance of 
machine learning algorithms such as Decision trees and Random Forest. 

## Dataset
The dataset consists of 1 year of historical transactional data and fraud flags. Dataset contains following columns:

transactionTime	 - The time the transaction was requested.
eventId - 	A unique identifying string for this transaction
accountNumber	 - The account number which makes the transaction
merchantId	- A unique identifying string for this merchant
mcc	- The merchant category code of the merchant
transactionAmount	The value of the transaction in GBP
posEntryMode -	The Point Of Sale entry mode
availableCash	- The (rounded) amount available to spend prior to the transaction
merchantCountry	- A unique identifying string for the merchant's country
merchantZip	- A truncated zip code for the merchant's postal region
	
posEntryMode Values	
00	Entry Mode Unknown
01	POS Entry Mode Manual
02	POS Entry Model Partial MSG Stripe
05	POS Entry Circuit Card
07	RFID Chip (Chip card processed using chip)
80	Chip Fallback to Magnetic Stripe
81	POS Entry E-Commerce
90	POS Entry Full Magnetic Stripe Read
91	POS Entry Circuit Card Partial




Note that for copyright reasons, I cannot publicly share the dataset. Thank you for your understanding. 
