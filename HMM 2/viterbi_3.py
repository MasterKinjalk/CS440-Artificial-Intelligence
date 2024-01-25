import math
from collections import defaultdict, Counter


class trelNode:
    def __init__(self, p, ptr, tag, w):
        self.p = p
        self.bkptr = ptr
        self.tag = tag
        self.word = w



import re
from collections import defaultdict, Counter

def countTags(sentences):
    tagWordCounts = defaultdict(Counter)  # Use defaultdict to eliminate the need for checking if a word exists
    countOfTags = Counter()
    IniTagCount = Counter()
    countTagTrans = defaultdict(Counter)  # Use defaultdict to eliminate the need for checking if a tag exists
    hapxCounts = Counter()
    hapxTC = 0
    uniqWord = set()

    for sentence in sentences:
        # Use itertools to efficiently iterate over pairs of words and tags
        for (word, tag), (_, nextTag) in zip(sentence, sentence[1:]):
            tagWordCounts[word].update([tag])  # Update tagWordCounts for the current word

            # Update countTagTrans for transitions from the current tag to the next tag
            countTagTrans[tag].update([nextTag])

            countOfTags.update([tag])  # Update countOfTags for the current tag

            if sentence.index((word, tag)) == 0:
                IniTagCount.update([tag])  # Update IniTagCount for the first tag in the sentence

    # Use a set to keep track of unique word-tag pairs for each tag
    uniqWordTagPairs = defaultdict(set)
    for sentence in sentences:
        for (word, tag) in sentence:
            uniqWordTagPairs[tag].add(word)
            uniqWord.add(word)

    # Add the 'UNKWORD' key to tagWordCounts with an empty Counter
    tagWordCounts['UNKWORD'] = Counter()
    hapxWord = []
    for tag in countOfTags:
        for word in uniqWordTagPairs[tag]:
            if tagWordCounts[word][tag] == 1:
                hapxWord.append(word)
                if '-' in word:
                    tagWordCounts["X-HYPHEN"][tag] += 0.5
                    continue
                if '$' in word:
                    tagWordCounts["X-DOLLAR"][tag] += 0.5
                    continue
                if "'" in word:
                    tagWordCounts["X-APOS"][tag] += 0.5
                    continue
                if word.endswith('ly'):
                    tagWordCounts['X-LY'][tag] += 0.5
                    continue
                if word.endswith('ing'):
                    tagWordCounts["X-ING"][tag] += 0.5
                    continue
                if word.endswith('ed'):
                    tagWordCounts["X-ED"][tag] += 0.5
                    continue
                if word.endswith('er'):
                    tagWordCounts["X-ER"][tag] += 0.5
                    continue
                if word.endswith('ic'):
                    tagWordCounts["X-IC"][tag] += 0.5
                    continue   
                if word.endswith('y'):
                    tagWordCounts["X-Y"][tag] += 0.5
                    continue  
                if word.startswith('re'):
                    tagWordCounts["X-RE"][tag] += 0.5
                if re.search(r'\d',word):
                    tagWordCounts["X-DIG"][tag] += 0.5
                    continue      
                hapxCounts.update([tag])
                hapxTC += 1

    return tagWordCounts, countOfTags, IniTagCount, countTagTrans, hapxCounts, hapxTC,hapxWord,uniqWord


def calculateProbabilities(train, tagWordCounts, countOfTags, IniTagCount, countTagTrans, hapxCounts, hapxTC, hapxWord,uniqWord):
    emissionProb = dict()
    transitionProb = dict()
    initialProb = dict()
    hapxProb = dict()

    totalUniqWords = len(tagWordCounts)     # Number of unique words in the training set
    totalTags = len(countOfTags) + 1           # Number of tags in the training set, + 1 to account for 'UNKNOWNTAG'


    alpha = 1  # Smoothing Constant
    # hapxProb builder
    for tag in countOfTags:
        hapxProb[tag] = (hapxCounts[tag] + alpha) / (hapxTC + alpha * totalTags)

    # emissionProb builder
    for word in tagWordCounts:
        emissionProb[word] = dict()
        for tag in countTagTrans:
            emissionProb[word][tag] = (tagWordCounts[word][tag] + (alpha * hapxProb[tag])) / (countOfTags[tag] + ( (alpha * hapxProb[tag]) * totalUniqWords))  # Numerator will be alpha for 'UNKNOWN'


    # transitionProb builder
    for prevTag in countTagTrans:
        transitionProb[prevTag] = dict()
        for nxtTag in countTagTrans:
            transitionProb[prevTag][nxtTag] = (countTagTrans[prevTag][nxtTag] + alpha) / (countOfTags[prevTag] + alpha * totalTags)

    # initialProb builder
    for tag in countTagTrans:
        initialProb[tag] = (IniTagCount[tag] + alpha) / (len(train) + alpha * totalTags)

    return emissionProb, transitionProb, initialProb

# def trellisAndBacktrack(test, tagWordCounts, emissionProb, transitionProb, initialProb):
#     predicts = []
#     com_suf = {'y': 'X-Y', 'er': 'X-ER', 'ing': 'X-ING', 'ly': 'X-LY', 'ic': 'X-IC', 'ed': 'X-ED'}
#     # Iterate over sentences in the test set
#     for sentence in test:
#         # Initialize previous word-tag nodes
#         previousWordTagNode = []

#         # Iterate over words in the sentence
#         for idx, word in enumerate(sentence):
#             # Initialize current word-tag nodes
#             NodesCurrentWordTag = []

#             # Check if the word is in tagWordCounts, and if not, it's 'UNKWORD'
#             if word in tagWordCounts:
#                 for tag in tagWordCounts[word]:
#                     maxNode = max(previousWordTagNode, key=lambda node: node.p + math.log(transitionProb[node.tag][tag]))  # for back pointer
#                     NodesCurrentWordTag.append(trelNode(math.log(emissionProb[word][tag]) + maxNode.p + math.log(transitionProb[maxNode.tag][tag]), maxNode, tag, word))
#             else:
#                 # Handle 'UNKWORD' with prefixes and suffixes
#                 unk_node = None
#                 for tag in transitionProb:
#                     maxNode = max(previousWordTagNode, key=lambda node: node.p + math.log(transitionProb[node.tag][tag]))  # for back pointer

#                     # Check if the word starts with any of the common prefixes
#                     if word.startswith('re'):
#                         unk_node = trelNode(math.log(emissionProb['X-RE'][tag]) + maxNode.p + math.log(transitionProb[maxNode.tag][tag]), maxNode, tag, word)
#                         break

#                     # Check if the word ends with any of the common suffixes
#                     if unk_node is None:
#                         for s in com_suf:
#                             if word.endswith(s):
#                                 unk_node = trelNode(math.log(emissionProb[com_suf[s]][tag]) + maxNode.p + math.log(transitionProb[maxNode.tag][tag]), maxNode, tag, word)
#                                 break

#                     if unk_node is None:
#                         unk_node = trelNode(math.log(emissionProb['UNKWORD'][tag]) + maxNode.p + math.log(transitionProb[maxNode.tag][tag]), maxNode, tag, word)

#                     NodesCurrentWordTag.append(unk_node)

#             previousWordTagNode = NodesCurrentWordTag

#         # Backtracking for each sentence
#         tempRevSen = []
#         CurrentNode = max(previousWordTagNode, key=lambda node_: node_.p)

#         while CurrentNode is not None:
#             tempRevSen.append((CurrentNode.word, CurrentNode.tag))
#             CurrentNode = CurrentNode.bkptr

#         tempRevSen.reverse()

#         predicts.append(tempRevSen)

#     return predicts

def trellisAndBacktrack(test, tagWordCounts, emissionProb, transitionProb, initialProb):
    predicts = []

    # Iterate over sentences in the test set
    for sentence in test:
        # Initialize previous word-tag nodes
        previousWordTagNode = []

        # Iterate over words in the sentence
        for idx, word in enumerate(sentence):
            # Initialize current word-tag nodes
            NodesCurrentWordTag = []

            if word in tagWordCounts:
                if idx == 0:
                    for tag in tagWordCounts[word]:
                        p = math.log(initialProb[tag]) + math.log(emissionProb[word][tag])
                        NodesCurrentWordTag.append(trelNode(p, None, tag, word))
                else:
                    for tag in tagWordCounts[word]:
                        p = math.log(emissionProb[word][tag]) + max([node.p + math.log(transitionProb[node.tag][tag]) for node in previousWordTagNode])
                        maxNode = max(previousWordTagNode, key=lambda node: node.p + math.log(transitionProb[node.tag][tag]))  # for back pointer
                        NodesCurrentWordTag.append(trelNode(p, maxNode, tag, word))
            else:
                word1 = word            # Add the newly defined word types
                if word1.endswith('ly'):
                    word1 = 'X-LY'
                elif word1.endswith('ing'):
                    word1 = 'X-ING'
                elif word1.endswith('ed'):
                    word1 = 'X-ED'
                elif word1.endswith('er'):
                    word1 = 'X-ER'
                elif word1.endswith('ic'):
                    word1 = 'X-IC'
                elif word1.endswith('y'):
                    word1 = 'X-Y'
                elif word1.startswith('re'):
                    word1 = 'X-RE'
                elif '-' in word1:
                    word1 = 'X-HYPHEN'
                elif "'" in word1:
                    word1 = 'X-APOS'
                elif '$' in word1:
                    word1 = 'X-DOLLAR'
                elif re.search(r'\d',word):
                    word1 = 'X-DIG'
                else:
                    word1 = 'UNKWORD'

                if idx == 0:
                    for tag in transitionProb:

                        p = math.log(initialProb[tag]) + math.log(emissionProb[word1][tag])
                        NodesCurrentWordTag.append(trelNode(p, None, tag, word))
                else:
                    for tag in transitionProb:
                        p = math.log(emissionProb[word1][tag]) + max([node.p + math.log(transitionProb[node.tag][tag]) for node in previousWordTagNode])
                        maxNode = max(previousWordTagNode, key=lambda node: node.p + math.log(transitionProb[node.tag][tag]))  # for back pointer
                        NodesCurrentWordTag.append(trelNode(p, maxNode, tag, word))

            previousWordTagNode = NodesCurrentWordTag

        # Backtracking for each sentence
        tempRevSen = []
        CurrentNode = max(previousWordTagNode, key=lambda node_: node_.p)

        while CurrentNode is not None:
            tempRevSen.append((CurrentNode.word, CurrentNode.tag))
            CurrentNode = CurrentNode.bkptr

        tempRevSen.reverse()

        predicts.append(tempRevSen)

    return predicts




def viterbi_3(train, test):
    tagWordCounts, countOfTags, IniTagCount, countTagTrans, hapxCounts, hapxTC, hapxWord,unword = countTags(train)
    
    emissionProb, transitionProb, initialProb = calculateProbabilities(train, tagWordCounts, countOfTags, IniTagCount, countTagTrans, hapxCounts, hapxTC, hapxWord,unword)

    predicts = trellisAndBacktrack(test, tagWordCounts, emissionProb, transitionProb, initialProb)

    return predicts
