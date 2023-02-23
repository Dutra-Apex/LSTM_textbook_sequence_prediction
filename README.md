# LSTM Textbook Sequence Prediction

The goal of this repo is to model a LSTM network to predict the optimal reading sequence of a given topic among a list of textbooks.
The 'optimal' reading list is defined by the term frequency-inverse document frequency of a given term.

For example, given the topic "preassure" among science textbooks (biology, chemistry and physics), the LSTM should be able to return a relevant list of chapters that correlate all the 3 fields through the topic. The reading list should be a curriculum candidate for the teaching of that topic. 
