"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
def train_baseline(train):
    # Initialize dictionaries to store word-tag counts and total tag counts
    CountWordTags = {}
    TotalWordTagCount = {}

    # Loop through each sentence in the training data
    for sen in train:
        # For each word-tag pair in the sentence
        for w, tag in sen:
            # Increment the count of this word-tag pair
            CountWordTags[(w, tag)] = CountWordTags.get((w, tag), 0) + 1
            # Increment the count of this tag
            TotalWordTagCount[tag] = TotalWordTagCount.get(tag, 0) + 1

    # Return the count dictionaries
    return CountWordTags, TotalWordTagCount


def find_most_common_tags(word_tag_counts: dict()):
    # Initialize a dictionary to store the most common tag for each word
    MostCommonTags = dict()

    # Loop through each word-tag pair and its count in the word-tag counts dictionary
    for (w, _), count in word_tag_counts.items():
        # If this word is not in the most common tags dictionary or 
        # if the count of this word-tag pair is greater than the count of the current most common tag for this word
        if w not in MostCommonTags or count > word_tag_counts[(w, MostCommonTags[w])]:
            # Update the most common tag for this word
            MostCommonTags[w] = _

    # Return the dictionary of most common tags
    return MostCommonTags

def predict_tags(test, most_common_tags, most_common_overall_tag):
    # Initialize a list to store the predicted tags for each sentence in the test data
    PredictionRes = []

    # Loop through each sentence in the test data
    for sen in test:
        # Predict the tags for each word in the sentence using the most common tags dictionary and 
        # the most common overall tag (used as a default when a word is not in the dictionary)
        PredicdtedSen= [(w, most_common_tags.get(w, most_common_overall_tag)) for w in sen]

        # Add the predicted tags for this sentence to the result list
        PredictionRes += [PredicdtedSen]

    # Return the list of predicted tags for each sentence in the test data
    return PredictionRes


def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # Train the baseline model on the training data
    CountWordTags, TotalWordTagCount = train_baseline(train)

    # Find the most common tag for each word based on the training data
    MostCommonTags = find_most_common_tags(CountWordTags)

    # Find the overall most common tag based on the training data
    MostCommonOverallTag = max(TotalWordTagCount, key=lambda tag: TotalWordTagCount[tag])

    # Predict tags for test data using trained model and return predictions
    return predict_tags(test, MostCommonTags, MostCommonOverallTag)