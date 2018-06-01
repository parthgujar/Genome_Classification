import numpy as np
import sys
import util

gene_len = 40

if len(sys.argv) < 5:
    sys.exit("gene_snippet_generator takes 5 arguments.")
try:
    num_snippets = int(sys.argv[1])
    mutation_rate = float(sys.argv[2])
    from_ends = int(sys.argv[3])
    output_file =  sys.argv[4]
except:
    sys.exit("invalid argument !")

output = ''
letters = ['A','B','C','D']

for i in range(num_snippets):
    first_rand=np.random.randint(0, 4, gene_len/2)  #generate first half
    # randomly
    #create a second half perfect palinfrom
    #then do the mutations
    gene_code = []
    gene_code_match = []
    for element in first_rand: # Generates letters for first 20 chars
        gene_code.append(letters[element])
    #generating the second half to make the string palindrome
    for i in range(len(gene_code)):
        gene_code_match.append(util.get_correct_match(gene_code[i]))
    gene_code_str = ''.join(gene_code) + ''.join(reversed(gene_code_match))
    for j in range(from_ends,len(gene_code_str)-from_ends):# randomly picks
                                                            # a letter
        random_int = np.random.randint(0,4)
        if j<0: j=j+len(gene_code_str) #getting rid of negative index
            # becaasue it is messing up the below replacment
        gene_code_str = gene_code_str[:j] + letters[random_int]+\
                                            gene_code_str[j+1:]
    for j in range(-from_ends,from_ends): # generates letters from ends till
        # first and last sections when within the distance from start and end
        random_float = np.random.uniform()
        if random_float <= mutation_rate :
            letter = util.mutation(gene_code_str[j])
            if j<0: j=j+len(gene_code_str) #replacing the negative index
                        #with the equivalant positive one
            gene_code_str = gene_code_str[:j] + letter + gene_code_str[j+1:]
    output = output + gene_code_str+'\n'

F = open(output_file,'w')
F.write(''.join(output))
F.close()
