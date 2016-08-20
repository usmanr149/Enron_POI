#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    """ given an opened email file f, parse 
    	out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming
        capabilities)
        and return a string that contains 
        all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning 
    ### of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(
        	bytes.maketrans(b"", b"") )
        #This removes the punctuation
        '''
        this maps every character from
		# string.punctuation to None. 
		Initialize a translation object from it.
        '''
        translator = str.maketrans({key: 
        	None for key in 
        	string.punctuation})
        '''
        text_string = content[1].translate(
        	string.maketrans("", ""), 
        	string.punctuation)
        '''
        ### project part 2: 
        ### comment out the line below
        words = text_string.translate(translator)
        stemmer = SnowballStemmer("english")
        word_list = words.split()
        words = [stemmer.stem(word) for word in word_list]
        final_words = ""
        final_words = " ".join(words)
        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)


    return final_words

    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print (text)



if __name__ == '__main__':
    main()

