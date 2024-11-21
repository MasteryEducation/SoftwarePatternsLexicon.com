---
linkTitle: "User-Based Collaborative Filtering"
title: "User-Based Collaborative Filtering: Recommending items based on the preferences of similar users"
description: "A design pattern for recommending items to users based on the preferences of other users who have similar tastes."
categories:
- Model Training Patterns
tags:
- Collaborative Filtering
- Recommender Systems
- User Similarity
- Machine Learning
- Personalization
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-training-patterns/collaborative-filtering/user-based-collaborating-filtering"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

User-Based Collaborative Filtering (UBCF) is a recommendation system design pattern that suggests items to a user based on the preferences of users with similar tastes. This technique leverages user-to-user similarity to predict the interests of a target user by considering users with the most comparable preferences to provide personalized recommendations.

## Concept and Workflow

The core idea in UBCF is to calculate the similarity between users based on their historical interaction data, such as ratings or purchase histories. Here’s a detailed workflow:

1. **Data Collection**: Gather historical data of users' interactions with items.
2. **Similarity Computation**: Calculate similarity metrics (e.g., Pearson correlation, cosine similarity) between users.
3. **Neighborhood Selection**: Identify a set of users (neighbors) who have the highest similarity scores with the target user.
4. **Prediction**: Aggregate the preferences of the neighbors to predict the target user’s preferences.
5. **Recommendation**: Recommend the top N items with the highest predicted preferences that the user has not interacted with.

## Example Implementation

### Python Example using `scikit-learn`
Here’s a simple example using `scikit-learn` to implement UBCF:

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'User': ['U1', 'U2', 'U3', 'U4', 'U5'],
    'Item1': [5, 4, np.nan, 2, 1],
    'Item2': [1, 2, 3, 4, 5],
    'Item3': [2, np.nan, 4, np.nan, 5],
    'Item4': [np.nan, 4, 5, 3, 2],
    'Item5': [1, 2, 1, 3, np.nan]
}

df = pd.DataFrame(data).set_index('User')
print(df)

user_similarity = cosine_similarity(df.fillna(0))
similarity_df = pd.DataFrame(user_similarity, index=df.index, columns=df.index)
print(similarity_df)

def predict_ratings(user_id, item_id):
    # Extract the relevant user similarities
    user_row = similarity_df.loc[user_id, :]
    similar_users = user_row.sort_values(ascending=False).index

    numerator, denominator = 0, 0
    for similar_user in similar_users:
        if not np.isnan(df.loc[similar_user, item_id]):
            user_similarity_score = user_row[similar_user]
            numerator += user_similarity_score * df.loc[similar_user, item_id]
            denominator += abs(user_similarity_score)
    
    return numerator / denominator if denominator != 0 else np.nan

predicted_rating = predict_ratings('U1', 'Item3')
print(f"Predicted rating for U1 on Item3: {predicted_rating}")
```

### R Example using `recommenderlab`
```R
library(recommenderlab)

data <- matrix(c(
  5, 1, 2, NA, 1,
  4, 2, NA, 4, 2,
  NA, 3, 4, 5, 1,
  2, 4, NA, 3, 3,
  1, 5, 5, 2, NA
), nrow=5, ncol=5, byrow=TRUE)
rownames(data) <- c("U1", "U2", "U3", "U4", "U5")
colnames(data) <- c("Item1", "Item2", "Item3", "Item4", "Item5")

r <- as(data, "realRatingMatrix")

recommender <- Recommender(r, method = "UBCF", param=list(method="Cosine", nn=3))
recommendations <- predict(recommender, r[1,], type="ratings")

as(recommendations, "matrix")
```

## Related Design Patterns

### Item-Based Collaborative Filtering
**Description**: Recommends items by comparing the similarities between items rather than users. Here the item similarities are used to predict a user's rating by considering the ratings of similar items.

### Matrix Factorization
**Description**: A method to reduce the dimensionality of user-item interaction data by decomposing it into lower-dimensional matrices. These matrices can reveal latent features explaining user preferences.

### Content-Based Filtering
**Description**: Recommends items based on the content or features of the items themselves. It uses similarities between items' content rather than users' preferences.

## Additional Resources

- [Recommender Systems Handbook, 2nd Edition by Francesco Ricci](https://www.springer.com/gp/book/9781489976369)
- [Machine Learning for Recommender Systems by Bernd Markus Hoeffgen](https://link.springer.com/book/10.1007/978-3-319-29659-3)

## Summary

User-Based Collaborative Filtering is a significant and straightforward approach for building recommender systems. By leveraging the preferences of similar users, it facilitates meaningful and personalized recommendations. Nonetheless, it faces scalability challenges as the number of users grows, and it might struggle with sparse datasets where users have few overlapping reviews.

Understanding UBCF alongside similar design patterns is essential for selecting the most suitable recommendation strategy for specific applications and ensuring efficient and effective deployments.
