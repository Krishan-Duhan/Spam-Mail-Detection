# Spam-Mail-Detection
Text Classification using Naive Bayes & Logistic Regression.

 >> Download the  spam/ham  (ham is  not spam)  dataset  availableon  the elearning.  The data set is divided into two sets: training set and test set. The dataset was used in the Metsis et al. paper [1]. Each set has two directories: spam and ham. All files in the spam folders are spam messages and all files in the ham folder are legitimate (non spam) messages.
 
>> Implement  the  multinomial  Naive  Bayes  algorithm  for  text  classification  described here: http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf(see   Figure   13.2).   Note   that   the algorithm uses add-one Laplace smoothing. Make sure that you do all the calculations in log-scale to avoid underflow. Use your algorithm to learn from the training set and report accuracy on the test set.

>> Implement  the  MCAP  Logistic  Regression  algorithm  with  L2  regularization  that  we discussed in class (see Mitchell's new book chapter). Try different values of λ. Use your algorithm to learn from the training set and report accuracy on the test set for different values ofλ. Use gradient  ascent  for  learning  the  weights.  Do  not  run  gradient  ascent  until  convergence;  you should put a suitable hard limit on the number of iterations.

>> Improved Naive Bayes and Logistic Regression algorithms by throwingaway (i.e., filtering out) stop words such as \the" \of" and \for" from allthe documents. A list of stop words can be found here: https://www.ranks.nl/stopwords. Report accuracy for both Naïve Bayes and Logistic Regression for this filtered set. Does the accuracyimprove? Explain why the accuracy improves or why it does not?

>> Compile file for compiling and executing my code. 
Report file that reports  the  accuracy  obtained  on  the  test  set, parameters used  (e.g.,  values ofλ,  hard  limit  on the  number of  iterations,  etc.).

>> References[1] V. Metsis, I. Androutsopoulos and G. Paliouras, \Spam Filtering with NaiveBayes -Which Naive Bayes?". Proceedings of the 3rd Conference on Email andAnti-Spam (CEAS 2006), Mountain View, CA, USA, 2006.
