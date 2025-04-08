import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X = {chr(i): 0 for i in range(ord('A'), ord('Z') + 1)}
    with open(filename, encoding='utf-8') as f:
        for line in f:
            for char in line.upper():
                if 'A' <= char <= 'Z':
                    X[char] += 1
    
    return X
# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!

def compute_log_probability(X, p, prior):
    F_y = math.log(prior) + sum(X[chr(i + ord('A'))] * math.log(p[i]) if p[i] > 0 else 0 for i in range(26))
    return F_y

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 hw2.py [letter file] [english prior] [spanish prior]")
        return
    
    letter_file = sys.argv[1]
    prior_english = float(sys.argv[2])
    prior_spanish = float(sys.argv[3])
    
    X = shred(letter_file)
    e, s = get_parameter_vectors()
    
    print("Q1")
    for char in sorted(X.keys()):
        print(f"{char} {X[char]}")
    
    print("Q2")
    print(f"{X['A'] * math.log(e[0]) if e[0] > 0 else 0:.4f}")
    print(f"{X['A'] * math.log(s[0]) if s[0] > 0 else 0:.4f}")
    
    print("Q3")
    F_english = compute_log_probability(X, e, prior_english)
    F_spanish = compute_log_probability(X, s, prior_spanish)
    print(f"{F_english:.4f}")
    print(f"{F_spanish:.4f}")
    
    print("Q4")
    diff = F_spanish - F_english
    if diff >= 100:
        print("0.0000")
    elif diff <= -100:
        print("1.0000")
    else:
        prob_english = 1 / (1 + math.exp(diff))
        print(f"{prob_english:.4f}")

if __name__ == "__main__":
    main()
