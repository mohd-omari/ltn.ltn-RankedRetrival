import glob
import lib as lb


'''preprocessing the text then tokenize it '''
processed_text=[]
for file in glob.glob("*.txt"): #make sure that documents are in the same folder with this py file
    print(file)
    fname = file
    file = open(file , "r")
    text = file.read().strip()
    file.close()
    
    processed_text.append(lb.word_tokenize(str(lb.preprocess(text))))
    
    
'''docs converted to uniqe sets of words'''
unique_text = []
for lis in  processed_text:
    unique_text.append(set(lis))


'''document freq of every word in the collection of docs'''
n = len(processed_text) #or len(unique_text)
df = {}
for i in range(n):
    tokens = unique_text[i]
    for w in tokens:
        if w in df.keys() : 
            df[w] = df[w]+1
        else:
            df[w] = 1
            
            
  
num_vocabs = len(df)
all_vocab = [x for x in df] #list of all words in the collection of docs 

'''for each doc, calculate tf_idf of each one of its words'''
doc = 0
tf_idf = {} 
# tf_idf will look like this  { doc no. : [(word,score) , (word,score) ,....] }
#                             { _ key _ : ___________ v a l u e s ___________ }

for i in range(n):
    tokens = processed_text[i]
    tok_counts = lb.Counter(tokens) #count of each word in the list (the doc)
    # print(tok_counts)
    for token in set(tokens):                        # for each word in the doc 
        tf = 1 + lb.np.log(tok_counts[token])          # 1 + log ( tf )
        df1 = df[token]                                # get df for the word 
        idf = lb.np.log((n)/(df1))                     # calculate idf for the word log ( no of docs / word's doc freq )
        
        if doc in tf_idf.keys():
         tf_idf[doc].append( (token,tf*idf) )
        else:
         tf_idf[doc] = [(token,tf*idf)]
         
    doc += 1
    

print("enter your query : ")
query = input()
print()
query = lb.word_tokenize(str(lb.preprocess(query))) #preprocessing query then tokenize it 

'''calculate tf_idf of each word in the query'''
tf_idf_q = {}      # tf_idf_q will look like this { "query" : [(word,score) , (word,score) ,....] }

tok_counts_q = lb.Counter(query) #count of each word in the query
# print(tok_counts_q)
for token_q in set(query):
 tf = 1 + lb.np.log(tok_counts_q[token_q])    # 1 + log ( tf )
 if token_q in df.keys():                     # if the word is in one of the docs: ok ,else 0
  df1 = df[token_q]                           # get df for the word
  idf = lb.np.log((n)/(df1))                  # calculate idf for the word
 else :
  idf= 0
        
 if "query" in tf_idf_q.keys():
  tf_idf_q["query"].append( (token_q,tf*idf) )
 else:
  tf_idf_q["query"] = [(token_q,tf*idf)]
 
  
#remark:  tf_idf = { doc no. : [(word,score) , (word,score) ,....] } 
#                  { - key - : --------- v a l u e s ------------- }
#         tf_idf_q={ "query" : [(word,score) , (word,score) ,....] }

final_res = {}

# final_res will look like this :
#{ (doc no. , word) : (doc no. ,word score in doc , word score in query ,word score in doc*word score in query ) }
#{  ----- key ----- : ---------------------------------- v a l u e s ------------------------------------------- }

doc=0
for val in tf_idf.values():
 for tup in val:
  for i in range(len(tf_idf_q["query"])):   # no. of loops = no. of query words
   if tf_idf_q["query"][i][0] in tup :      # if the query word is in that doc :
    final_res[(doc,tup[0])] = (doc,tup[1],tf_idf_q["query"][i][1],tup[1]*tf_idf_q["query"][i][1])
 doc+=1
 


similarity={}
temp = 0.0000000000001                     
for i in final_res.keys() :                #get the tuple i, i = (doc no,word)
  if i[0] in similarity.keys():         
      similarity[i[0]]+=final_res[i][3]    #if the doc no. was already a key of similarity dict. , add-up the score of all its words
      similarity[i[0]]+=temp
  else :
      similarity[i[0]]=final_res[i][3]
      similarity[i[0]]+=temp
  temp+=0.0000000000001
"""next we will swap similarity dictionary keys by its values , so if two keys have the same values
the dictionary will add one of them as the new key and discard the other , so we add a very small
value -that doesn't make a difference- to the similarity dictionary values before swapping"""

final_rank = dict(zip(similarity.values(),similarity.keys()))            #swapping
final_rank = lb.OrderedDict(sorted(final_rank.items(), reverse=True))

rank = 1
for k, v in final_rank.items():
    print("rank: ",rank,"  doc: ", v, "     score: ",round(k,4))
    print()
    print(" ".join(processed_text[v]))
    print()
    print()
    rank+=1



     
 
      






