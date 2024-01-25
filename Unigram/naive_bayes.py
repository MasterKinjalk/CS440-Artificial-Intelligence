# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023

import reader
import math
from tqdm import tqdm
from collections import Counter

stopword = ['later', 'please', 'with', 'there', 'less', 'use', 'come', 'per', 'took', 'bj', 'v', 'n2', 'c2', 'cg', "you've", 'hasnt', 'rr', 'ad', 'never', 'certain', 'whod', 'oh', 'amount', "it's", 'nor', 'ni', 'dt', 'ih', 'hes', "can't", 'ibid', 'fix', 'gy', 'f', 'considering', 'while', 'added', 'ae', 'pas', 'unlikely', 'truly', 'changes', 'sq', 'are', 'suggest', 'hasn', 'quickly', 'usefulness', 'werent', 'asking', 'behind', 'similarly', 'part', "when's", 'want', 'ie', 'don', 'far', 'que', 'm', 'par', 'everybody', 'e2', 'os', 'stop', 'com', 'fn', 've', 'u', 'going', "hasn't", 'have', 'an', 'd', 'little', 'mo', 'allow', 's2', 'tends', 'clearly', 'mean', 'oo', 'gl', 'fj', 'did', 'particularly', 'concerning', 'whomever', 'beforehand', 'fl', 'miss', 'rt', 'un', "here's", 'ke', 'followed', 'sometimes', 'actually', 'whatever', 'yes', 'anywhere', 'thereby', 'somethan', 'jr', 'where', 'jt', 'a4', 'outside', 'thin', 'respectively', 'ip', 'wasn', 'better', 'ox', 'vq', 'ng', 'forth', "there's", 'da', "a's", "shouldn't", "it'd", 'into', "what's", 'least', 'ar', 'em', 'elsewhere', 'y2', 'accordingly', 'cannot', 'vd', 'cit', 'pn', 'besides', 'iz', 'xt', "t's", 'ca', 'w', 'yourself', 'willing', 'c1', 'go', 'ts', 'et', 'ab', 'various', 'td', 'strongly', 'on', 'cj', 'cs', 'aj', 'recent', 'resulted', 'volumtype', 'therefore', 'beyond', "they'll", 'pf', 'gs', 'bu', 'mn', 'amongst', 'put', 'regards', 'seem', 'seeming', 'whence', 'liked', 'sa', 'ko', 'av', 'rv', 'already', 'pj', 'du', 'seemed', 'soon', "i'll", 'well', 'end', 'vol', 'uo', 'enough', 'heres', 'bill', 'ba', 'looking', "i've", 'sup', 'over', 'saw', 'dl', 'wasnt', 'went', 'plus', 'done', 'necessarily', 'same', 'i3', '0o', 'wouldn', 'downwards', 'ou', 'tell', 'cv', 'follows', 'example', 'known', 'y', 'ig', 'likely', 'what', 'welcome', "you'll", 'present', 'sl', 'meantime', 'al', "needn't", 'sub', 'lest', 'home', 'br', 'anybody', 'fa', 'inward', 'hadn', 'bn', 'everyone', 'shouldn', 'keeps', 'ue', 'somehow', 'haven', "shan't", 'him', 'throug', 'often', 'against', 'following', 'ups', 'been', 'la', 'only', "you'd", 'bk', 'biol', 'used', 'ee', 'zi', 'b2', 'neither', 'a2', 'ot', 'thered', 'any', 'latterly', 'will', 'research-articl', 'cz', 'keep', 'recently', 'nearly', 'df', "should've", 'useful', 'research', 'being', 'the', 'youre', 'g', 'nr', 'thats', 'might', 'ref', 'particular', 'another', "he'd", 'whom', 'thereof', 'rf', 'information', 'sure', 'furthermore', 'having', 'onto', 'twenty', 'au', 'uses', 'mt', 'h', 'whats', 'substantially', 'h2', 'oj', 'ed', 'unlike', 'though', 'rq', 'bx', 'look', 'really', 'oa', 'causes', 'described', 'brief', 'since', 'rn', "there've", 'nay', "didn't", 'may', 'her', 'ml', 'pk', 'tries', 'almost', 'last', 'second', 'sometime', 'merely', 'sorry', 'none', 'has', 'pages', 'accordance', 'ec', 'oc', 'sd', 'ms', 'whenever', 'hers', 'seven', "won't", 'con', "why's", 'using', 'tj', 'reasonably', 'en', 'something', 'possible', 'three', 'name', 'tx', 'after', 'hopefully', 'tb', 'ff', 'again', 'thus', "they're", 'front', 'tried', 'mrs', 'indeed', "she'll", 'edu', 'results', 'nothing', 'ds', 'ending', 'such', 'afterwards', 'except', 'af', 'ep', 'off', 'ap', 'next', 'cf', 'ought', 'ij', 'b3', 'affected', 'ea', 'og', 'his', 'significantly', 'qu', 'ju', 'during', 'goes', 'viz', 'fr', 'predominantly', "they've", 'el', "it'll", 'hs', 'more', 'it', 'c3', "she'd", 'awfully', 'found', '0s', 'secondly', 'thru', 'thoroughly', 'previously', 'see', 'adj', 'yt', 'seriously', 'mustn', 's', 'll', '3a', 'ry', 'these', 'xi', 'need', 'sj', 'unless', 'so', 'aside', 'try', 'tl', 'oi', 'iy', 'even', "there'll", 'words', 'ro', 'couldnt', 'dy', 'xx', 'va', 'a', 'your', 'them', 'ask', 'other', 'ru', 'thank', 'ones', 'third', "he'll", 'herein', 'wouldnt', 'rh', 'theyd', '3b', 'mainly', 'similar', 'whereby', 'pt', 't', 'xo', "wasn't", 'cp', 'got', 'oq', 'significant', 'sp', '6o', 'dx', 'mg', 'l2', 'somewhere', 'cn', 'but', 'meanwhile', 'this', 'anything', 'insofar', 'still', 'sm', 'noone', 'currently', 'am', 'dd', 'se', 'gr', 'how', 'former', 't2', 'immediate', 'ix', 'hello', 'p1', 'q', 'hereby', 'seen', 'consider', 'believe', 'those', 'few', 'e', 'page', "ain't", 'j', 'take', 'for', "doesn't", 'latter', 'whether', 'normally', 'full', 'widely', 'affects', 'somebody', 'can', 'fy', 'further', 'well-b', 'throughout', 'successfully', 'maybe', 'hh', 'ey', 'p', "we'd", 'gj', 'shall', 'else', 'gi', 'sixty', 'sf', 'shan', 'thereupon', 'uj', 'kj', 'nj', 'then', 'auth', 'showns', 'rm', 'bs', 'is', 'world', 'obtained', 'quite', 'st', "hadn't", 'date', 'dc', 'they', 'thoughh', 'hundred', 'whereafter', 'z', 'associated', 'show', 'dp', 'presumably', 'whoever', 'whim', 'nobody', 'way', 'until', 'like', 'az', '3d', 'ac', 'no', 'qv', 'ho', 'empty', 'fs', 'yet', 'entirely', 'from', 'sec', 'just', 'll', 'own', 'immediately', 'ah', 'bi', 'tm', 'find', 'hereupon', 'under', 'nt', 'won', 'io', 'rather', 'here', 'whole', 'twice', 'becomes', 'upon', 'howbeit', 'obviously', 'becoming', 'ag', 'nc', 'shows', 'toward', 'could', 'obtain', 'owing', 'appropriate', 'say', 'yours', 'anyways', 'relatively', 'x3', 'ri', 'placed', 'i2', 'otherwise', 'become', 'til', 'instead', 'pl', 'resulting', 'selves', 'above', 'taken', 'fifteen', 'near', 'thickv', "he's", 'thereafter', 'yr', 'uk', 'giving', 'taking', 'seeing', 'act', 'theirs', 'ob', 'sy', 'between', 'bl', 'also', 'mostly', 'ct', 'corresponding', 'pd', 'give', 'wants', 'aren', 'readily', 'once', 'consequently', 'some', 'important', 'pp', 'yl', 'thanks', 'ln', 'wheres', 'provides', 'ny', 'shown', 'si', 'necessary', 'l', 'lb', 'she', 'mug', 'yj', 'however', 'eq', 'made', 'invention', 'nine', 'ci', 'towards', 'course', 'cd', 'both', 'and', "don't", 'i7', 'should', 'tr', 'lets', 'cl', 'million', 'mightn', 'each', 'theyre', "where's", 'cause', 'eo', 'does', 'although', 'around', 'dr', 'cm', 'c', 'vt', 'ps', 'alone', 'together', 'ir', 'comes', 'ours', 'couldn', 'na', 'hj', 'eight', 'everywhere', 'ne', 'ax', 'new', 'ph', 'bc', 'wont', 'hr', '6b', "couldn't", 'hither', 'available', 'let', 'ef', 'im', 'abst', 'became', 'fo', 'nos', 'lf', 'kept', 'namely', 'ti', 'refs', 'whose', 'effect', 'od', 'whither', "she's", 'xf', "that's", 'means', 'o', "we've", 'happens', 'specified', 'fu', 'cr', 'etc', 'saying', 'out', 'e3', 'po', 'which', 'di', 'appear', 'ok', 'containing', 'hid', 'op', 'cy', 'tn', 'x2', 'know', 'my', 'n', 'had', 'allows', 'vols', 'ignored', 'gotten', 'b1', 'youd', 'i8', 'lt', 'zz', 'themselves', 'away', 'proud', 'h3', 'cc', 'th', 'ol', 'zero', 'okay', 'serious', 'wish', 'tt', 'dj', 'too', 'tf', 'er', 'ga', 'their', 'greetings', 'anyhow', 'says', 'exactly', 'ex', 'line', 'ourselves', 'aw', 'i6', 'formerly', 'certainly', 'within', "isn't", 'trying', 'eg', 'tc', 'eu', 'regarding', 'id', 'nowhere', 'isn', 'fc', 'ev', 'hu', "i'd", 'value', 'iv', 'past', "haven't", 'promptly', 'fill', 'rl', 'shed', 'r2', 'wherein', "that've", 'herself', 'lately', "c's", 'slightly', 'now', 'ej', 'usefully', 'old', "i'm", 'ltd', 'be', 'possibly', 'pu', 'ur', 'think', 'tv', 'i4', 'nn', 'omitted', "weren't", 'um', 'through', 'not', 'much', 'itd', 'ib', 'index', "c'mon", 'inc', 'bd', 'www', 'cry', 'affecting', 'showed', 'interest', 'didn', 'es', 'dk', 'ra', 'contains', 'at', "they'd", 'knows', 'five', 'thence', 'nl', 'vj', 'te', 'get', 'xk', 'do', "that'll", 'me', 'ut', 'he', 'ay', 'tq', 'related', 'km', 'nd', 'vu', 'lc', 'or', 'needs', 'you', 'make', 'mu', 't1', 'hy', 'sc', "mightn't", 'cx', "we'll", 'primarily', 'many', 'ge', 'oz', 'seems', 'indicates', 'all', 'de', 'ninety', 'rs', 'specifying', 'yourselves', 'said', 'ever', 'why', 'wo', 'six', 'sincere', 'usually', 'perhaps', 'up', 'most', 'via', 'beginnings', 'describe', 'apart', 'anymore', 'xv', 'therere', "wouldn't", 'x1', 'nevertheless', 'himself', 'qj', 'thousand', 'sensible', 'everything', 'always', 'moreover', 'rc', 'run', 'cq', 'ran', 'nonetheless', 'sent', 'py', 'call', 'detail', 'ys', 'poorly', 'help', 'shes', 'despite', 'noted', 't3', 'whereupon', 'm2', 'tip', 'very', 'we', 'able', 'sz', 'thereto', 'wherever', 'http', 've', 'contain', 'hed', 'who', 'importance', 'several', 'either', 'les', 'section', 'our', 'd2', 'because', 'potentially', 'hence', 'gets', 'iq', 'every', 'bottom', 'wi', 'given', 'ow', 'inasmuch', "aren't", 'sufficiently', "who'll", 'fi', 'pm', 'ao', 'vo', 'js', 'beside', 'sr', 'ic', 'unfortunately', 'indicated', 'gives', "how's", 'if', 'begin', 'doing', 'fify', 'pi', 'first', 'others', 'myself', 'thanx', 'to', 'wonder', 'among', 'back', 'arent', 'anyway', 'jj', 'et-al', 'wa', 'rd', 'unto', 'xl', 'tp', 'ii', 'down', "mustn't", 'los', 'pr', 'lr', 'largely', 'different', 'ce', 'getting', 'cu', 'lo', 'k', 'ia', 'before', 'was', 'apparently', 'eighty', 'ns', 'hereafter', 'fifth', 'along', 'thorough', 'whos', 'co', 'pq', 'u201d', 'mill', 'i', 'ei', 'est', 'ma', 'itself', 'twelve', 'side', 'probably', 'one', 'kg', 'bt', 'when', 'about', 'eleven', 'indicate', 'ten', 'announce', 're', 'especially', 'needn', 'its', 'f2', 'gone', 'arise', 'of', 'novel', 'pe', 'ui', 'makes', 'mine', 'in', 'without', 'best', 'le', 'than', 'forty', 'x', 'briefly', 'ain', 'due', 'non', 'p3', 'il', 'system', 'r', 'doesn', 'regardless', 'begins', 'hardly', 'somewhat', 'inner', 'that', 'two', 'bp', 'below', 'wed', "who's", 'anyone', 'theres', 'om', 'specify', 'by', 'therein', 'xj', 'b', 'must', 'gave', 'whereas', 'were', 'right', 'appreciate', 'xs', 'looks', 'across', "what'll", 'overall', 'us', 'four', 'weren', 'hi', 'pc', 'self', 'would', 'a1', 'vs', 'ft', 'thou', 'move', 'top', 'beginning', 'cant', 'ss', 'specifically', 'a3', 'fire', 'mr', "let's", 'someone', 'definitely', "you're", 'lj', 'as', 'xn', 'amoungst', 'approximately', 'rj', 'came', 'pagecount', 'ch', 'p2', 'sn', "we're", 'ls', 'according', 'ord']
'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naiveBayes(dev_set, train_set, train_labels, laplace=2.0, pos_prior=0.9, silently=False):
    print_values(laplace, pos_prior)

    terms = set()
    for rev in train_set:
        for w in rev:
            if w not in stopword:
                terms.add(w)       

    FinalResult = []

    wordcPos = Counter()
    wordcNeg = Counter()

    for i, rev in enumerate(train_set):
        label = train_labels[i]

        if label == 0:
            wordcNeg.update(rev)
        else:
            wordcPos.update(rev)

    totalWordsPos = sum(wordcPos.values())
    totalWordsNeg = sum(wordcNeg.values())

    prob_dict = {}
    for w in terms:
        laplaceProbPos = (wordcPos[w]+laplace)/(totalWordsPos+laplace*len(terms))
        laplaceProbNeg = (wordcNeg[w]+laplace)/(totalWordsNeg+laplace*len(terms))
        prob_dict[(w,0)]= math.log(laplaceProbNeg)
        prob_dict[(w,1)] = math.log(laplaceProbPos)
        
    pos_post = 1- pos_prior
    for sentence in dev_set:
        pos_prob = math.log(pos_prior)
        neg_prob = math.log(pos_post)
        for word in sentence:
            if (word,1) in prob_dict.keys():
                pos_prob += prob_dict[(word,1)]
            elif((word,1) not in prob_dict.keys()):
                pos_prob += math.log((laplace)/(totalWordsPos+laplace*len(terms)))
            if (word,0) in prob_dict.keys():
                neg_prob += prob_dict[(word,0)]
            else:
                neg_prob += math.log((laplace)/(totalWordsPos+laplace*len(terms)))

        if(pos_prob > neg_prob):
            FinalResult.append(1)
        else:
            FinalResult.append(0)    


    return FinalResult