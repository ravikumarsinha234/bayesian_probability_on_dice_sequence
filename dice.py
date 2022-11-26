import numpy as np
import sys
import itertools

# Check the command-line
if len(sys.argv) != 5:
    print(f"Usage: {sys.argv[0]} <dice> <sides> <input.txt> <out.csv>")
    exit(0)

# Get the command-line arguments
num_dice = int(sys.argv[1])
num_sides = int(sys.argv[2])
infilename = sys.argv[3]
outfilename = sys.argv[4]

# Bounds on the sum: sum_lb <= sum < sum_ub
sum_ub = num_dice * num_sides + 1
sum_lb = num_dice

# Pretty print the lookup table
def show_odds(odds):
    global sum_lb, sum_ub
    print('\n*** Lookup Table ***')
    for i in range(sum_lb, sum_ub):
        print(f"\tGiven d={i}:")
        print(f"\t\t", end="")
        for j in ['H','E','L']:
            print(f"P({j})={odds[j][i]:.5f} ", end="")
        print()

# Pretty print entries of 'array'
def show_array(label, array):
    global sum_lb, sum_ub
    print(f"\t{label}: ", end="")
    for i in range(sum_lb, sum_ub):
        print(f"{i}:{array[i]:.4f} ", end="")
    print()
    
# You will need a recursive function here to fill the how_many_ways array
# how_many_ways[8] holds number of ways n dice (each with m sides) make the number 8
def fill_how_many_ways(sides,dice,how_many_ways):
    for rolls in itertools.product(range(1,sides+1), repeat=dice):
        how_many_ways[sum(rolls)] += 1

print("*** Dice ***")
how_many_ways = np.zeros(sum_ub)
fill_how_many_ways(num_sides,num_dice,how_many_ways)
print(how_many_ways)
show_array('How many ways', how_many_ways)

# What is the probability of getting a sum of n when you roll the dice?
priors = how_many_ways/how_many_ways.sum()
show_array('Priors', priors)

# Create a lookup table
# Example: odds['H'][8] holds the probability of an 'H' given the original sum is 8
odds = {'H': np.zeros(sum_ub), 'E': np.zeros(sum_ub), 'L': np.zeros(sum_ub)}

# Fill in the lookup table
for i in range(sum_lb,sum_ub):
  odds['E'][i] = how_many_ways[i]/how_many_ways.sum()
  odds['H'][i] = how_many_ways[i+1:].sum()/how_many_ways.sum()
  odds['L'][i] = how_many_ways[:i].sum()/how_many_ways.sum()

# Show the lookup table
show_odds(odds)

# Open the output file
outfile = open(outfilename,'w')

# Write the header
outfile.write("input,")
for i in range(sum_lb, sum_ub):
    outfile.write(f"P(d={i}),")
outfile.write("guess\n")

# Open the input file
with open(infilename, 'r') as f:

    # Step through each line in the input file
    for row in f.readlines():

        # Remove newline
        row = row.strip()

        # Shorten long inputs to just 7 characters
        if len(row) <= 7:
            out_row = row
        else:
            out_row = row[:4] + "..."

        # Is this an empty line?
        if len(row) == 0:
            continue

        # print(f"\n*** Input:\"{row}\" ***")

        # Compute the likelihoods
        # Example: likelihoods[9] holds the probability of 'row' given the original sum is 9

        ## Your code here
        likelihoods = np.zeros(sum_ub)
        for i in range(sum_lb,sum_ub):
            likelihoods[i] = np.prod([odds[a][i] for a in row])
        # show_array("Likelihoods", likelihoods)

        # Bayes rule: the posterior is proportional to the likelihood times the prior
        unnormalized_posteriors = np.multiply(likelihoods,priors)
        # show_array("Unnormalized posteriors", unnormalized_posteriors)

        # But they don't sum to 1.0, so we need scale them so they do
        marginal_likelihood = np.sum(np.multiply(likelihoods,priors))
        # print(f"\tP({row}) = {marginal_probability:.7f}")
        normalized_posteriors = np.divide(unnormalized_posteriors,marginal_likelihood)
        # show_array("Normalized posteriors", normalized_posteriors)

        # Write them out
        outfile.write(f"{out_row},")
        for i in range(sum_lb, sum_ub):
            outfile.write(f"{normalized_posteriors[i]:.5f},")

        # Which is the most likely explanation?
        best = normalized_posteriors.argmax(axis=0)
        outfile.write(f"{best}\n")

# Close the output file
outfile.close()
