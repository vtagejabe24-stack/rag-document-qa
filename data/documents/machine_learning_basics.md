# Machine Learning Basics

Machine learning is a field of artificial intelligence in which algorithms learn patterns from data instead of being explicitly programmed with rules. A model is trained on examples and is then used to make predictions on new, unseen data.

## Supervised Learning

Supervised learning trains a model on labeled data, where every training example comes with the correct output. The goal is to learn a mapping from inputs to outputs. Classification predicts a discrete category, such as whether an email is spam. Regression predicts a continuous value, such as the price of a house. Common supervised algorithms include linear regression, logistic regression, decision trees, support vector machines and neural networks. The main advantage of supervised learning is that its performance is easy to measure, because predictions can be compared directly with the known labels. Its main disadvantage is the cost of collecting and labeling large datasets.

## Unsupervised Learning

Unsupervised learning works with unlabeled data and tries to discover hidden structure on its own. Clustering groups similar examples together, as in k-means, which assigns each point to the nearest of k cluster centers. Dimensionality reduction, for example principal component analysis (PCA), compresses data into fewer features while keeping most of the variation. Unsupervised methods are used for customer segmentation, anomaly detection and exploratory analysis.

## Reinforcement Learning

In reinforcement learning an agent interacts with an environment, takes actions and receives rewards or penalties. The agent learns a policy that maximizes the total cumulative reward over time. Reinforcement learning has been used to train game-playing systems and to control robots.

## Overfitting and Regularization

Overfitting happens when a model memorizes the training data, including its noise, and therefore performs poorly on new data. A typical sign is very low training error combined with high validation error. Underfitting is the opposite problem: the model is too simple to capture the pattern. Overfitting can be reduced by collecting more data, using a simpler model, applying regularization, using early stopping, or using cross-validation to choose hyperparameters. L2 regularization adds a penalty proportional to the squared size of the weights to the loss function, which discourages large weights.

## Evaluating Models

Accuracy is the fraction of predictions that are correct, but it is misleading when classes are imbalanced. Precision is the fraction of predicted positives that are truly positive. Recall is the fraction of actual positives that the model finds. The F1 score is the harmonic mean of precision and recall. Data is usually split into training, validation and test sets; the test set is used only once, at the end, to estimate real-world performance.
