"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect
alpha = 0.00001

from concurrent.futures import ThreadPoolExecutor
import math

def countTags(sentences):
    tagCount = Counter()
    tagPairCount = defaultdict(Counter)
    tagWordCount = defaultdict(Counter)
    
    for sentence in sentences:
        prevTag = None
        for word, tag in sentence:
            tagCount[tag] += 1
            if prevTag is not None:
                tagPairCount[prevTag][tag] += 1
            tagWordCount[tag][word] += 1
            prevTag = tag
    
    return tagCount, tagPairCount, tagWordCount

def calculateInitialProbabilities(tagCount, totalSentences, alpha):
    totalTags = len(tagCount)
    initialProb = {}
    
    for tag, count in tagCount.items():
        initialProb[tag] = (count + alpha) / (totalSentences + alpha * totalTags)
    
    return initialProb

def calculateEmissionProbabilities(tagCount, tagWordCount, alpha):
    emissionProb = defaultdict(lambda: defaultdict(lambda: 0))
    
    for tag, count in tagCount.items():
        uniqueWords = len(tagWordCount[tag])
        for word, wordCount in tagWordCount[tag].items():
            emissionProb[tag][word] = (wordCount + alpha) / (count + alpha * (uniqueWords + 1))
        emissionProb[tag]['UNKNOWN'] = alpha / (count + alpha * (uniqueWords + 1))
    
    return emissionProb

def calculateTransitionProbabilities(tagCount, tagPairCount, alpha):
    transitionProb = defaultdict(lambda: defaultdict(lambda: 0))
    
    for tag, count in tagCount.items():
        for nextTag in tagCount:
            transitionProb[tag][nextTag] = (tagPairCount[tag][nextTag] + alpha) / (count + alpha * (len(tagCount) + 1))
    
    return transitionProb

def training(sentences):
    totalSentences = len(sentences)
    
    with ThreadPoolExecutor() as executor:
        futureTagCounts = executor.submit(countTags, sentences)
        
        # Wait for the counting process to finish and retrieve the results
        tagCount, tagPairCount, tagWordCount = futureTagCounts.result()
    
    initialProb = calculateInitialProbabilities(tagCount, totalSentences, alpha)
    
    with ThreadPoolExecutor() as executor:
        futureEmissionProbs = executor.submit(calculateEmissionProbabilities, 
                                              tagCount,
                                              tagWordCount,
                                              alpha)
        
        futureTransitionProbs = executor.submit(calculateTransitionProbabilities,
                                                tagCount,
                                                tagPairCount,
                                                alpha)
        
        # Wait for the probability calculations to finish and retrieve the results
        emissionProb = futureEmissionProbs.result()
        transitionProb = futureTransitionProbs.result()

    return initialProb, emissionProb, transitionProb


# Function to calculate probabilities for the first word in the sentence
def calcFirstWordlProb(word, emitProb, initProb):
    logProbabilities = {}
    predictedTagSequence = {}
    
    for tag, emitProbTag in emitProb.items():
        logProbabilities[tag] = math.log(initProb[tag]) + math.log(emitProbTag.get(word, emitProbTag['UNKNOWN']))
        predictedTagSequence[tag] = [tag]
    
    return logProbabilities, predictedTagSequence

# Function to calculate probabilities for words other than the first one in the sentence
def calcOtherWordProb(word, prevProb, prevPredictTagSeq, emitProb, transProb, tagCounts, totalTags):
    logProbabilities = {}
    predictedTagSequence = {}
    
    for currentTag, emitProbCurrentTag in emitProb.items():
        maxProbability = float("-inf")
        optimalPrevTag = None
        currentEmitProb = math.log(emitProbCurrentTag.get(word, emitProbCurrentTag['UNKNOWN']))
        
        for prevTag, prevProbPrevTag in prevProb.items():
            transitionProbability = math.log(transProb[prevTag].get(currentTag, alpha / (tagCounts[prevTag] + alpha * (totalTags + 1))))
            
            currentProbability = prevProbPrevTag + transitionProbability + currentEmitProb
            
            if currentProbability > maxProbability:
                maxProbability = currentProbability
                optimalPrevTag = prevTag
        
        logProbabilities[currentTag] = maxProbability
        predictedTagSequence[currentTag] = prevPredictTagSeq[optimalPrevTag] + [currentTag]
    
    return logProbabilities, predictedTagSequence

# Main function to perform a step forward in the Viterbi algorithm
def viterbi_stepforward(i, word, prevProb, prevPredictTagSeq, initProb, emitProb, transProb, tagCounts, totalTags):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.   
    if i == 0:
        return  calcFirstWordlProb(word, emitProb, initProb)
    
    return calcOtherWordProb(word, prevProb, prevPredictTagSeq, emitProb, transProb, tagCounts, totalTags)
    

def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    tagCounts = defaultdict(int)

    for sentence in train:
        for _, tag in sentence:
            tagCounts[tag] += 1
    
    totalTags = len(tagCounts)

    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, init_prob, emit_prob, trans_prob, tagCounts, totalTags )

            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        # Find the best final tag and the best path in one step
        _, bestPath = max((tag, predict_tag_seq[tag]) for tag in log_prob)

        # Create the tagged sentence
        taggedSentence = list(zip(sentence, bestPath))

        # Append the tagged sentence to the predictions
        predicts.append(taggedSentence)

    print(len(predicts))
    return predicts




