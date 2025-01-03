Hello! Thanks for checking out my project Celebrity Matchmaking.

This project started out of my own curiousity. The website www.whosdatedwho.com is a 
user created and maintained website that has pages for many celebrities and their previous relationships.
I scraped some of the data off of this website to create a novel dataset of celebrity relationships,
including various personal features and relationship features. I was curious to see if I could
train a GNN model to somewhat accurately predict new relationships in a celebrity relationship social
network. So thats what I did. If you want to try out the code just downlaod and run python gnn.py.
This will train a graphSAGE model on all relationships before 2018 and test on relationships starting 
after 2018. If you want to use the dataset all of it is saved in the info folder which was gathered over
September 2024. 

This project is a work in process! Some of my next steps include:
    1. Using various explainability measures ex. LIME and SHAP to understand the why
    the model makes certain predictions
    2. Compare different methods for selecting negative edges for training and testing.
    Using randomly sampled negative edges should be fairly easy for the model to distinguish
    between positive and negative edges. I will try a few strategies for making more difficult
    negative edges and see how this affects model accuracy
    3. Look 