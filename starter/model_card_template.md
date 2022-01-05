# Model Card
This model card describes about the income prediction model trained on census income dataset.

## Model Details
The Scikit-Learn's Logistic Regression model is trained on the census income dataset.

## Intended Use
This model is created to prove that it's possible to predict a person's income level (>50k, <=50k) based on just a few characteristics of the person including demographics, occupation information.

## Training Data
The dataset contains 48842 salary-level instances with an associated person's 14 attributes. 80% are randomly chosen to train a model

## Evaluation Data
The dataset contains 48842 salary-level instances with an associated person's 14 attributes. 20% are randomly chosen to test the trained model's performance.

## Metrics
The model is evaluated on precision, recall and f1 scores.

## Ethical Considerations
The model is trained on a public dataset from UCI ML repository.
## Caveats and Recommendations
The model is just created for a demonstration purpose so it's not highly optimized for performance.