import random
from collections import Counter
import numpy as np 
import tqdm


def generate_random_sequence(n, lower=0, upper=1):
  """
  Generates a random sequence of n numbers within the specified range.

  Args:
      n: The length of the sequence.
      lower: The lower bound of the range (inclusive). Defaults to 0.
      upper: The upper bound of the range (inclusive). Defaults to 1.

  Returns:
      A sequence of n random integers from lower to upper inclusive, 
      The first five numbers of the sequence sorted first by descending frequency then ascending order.
  """

  seq = [random.randint(lower, upper) for _ in range(n)]
  #seq = np.array(seq)

  counted = Counter(seq)
    # Custom sorting function (key)
  def sort_key(item):
    return (-counted[item], item)  # Sort by decreasing frequency, then ascending value
  
  freq = sorted(seq, key=sort_key)
  first_five = []
  tracker = set({})

  for s in freq:
    if len(first_five) >= 5:
      break
    if s not in tracker:
      first_five.append(s)
      tracker.add(s)

  return seq, freq, first_five

import os
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)  
print(data_dir)
file_path = os.path.join(data_dir, 'sequences100.csv')
print(file_path)

#f = open("../data/sequences100.csv", "w")
#f.close()

files = [('sequences100_test.csv',100), ('sequences200_validate.csv',200), ('sequences500_test.csv',500), 
         ('sequences1000_train.csv',1000), ('sequences1000_validate.csv',1000), ('sequences5000_train.csv',5000)]

for filename in files:
  file_path = os.path.join(data_dir, filename[0])
  with open(file_path, 'w') as f:
    for _ in tqdm.tqdm(range(filename[1])):
      sequence, freq, first_five = generate_random_sequence(64, 1, 19)  # Generate a sequence of 64 bits (0 or 1)
      ss = np.array(sequence + first_five)
      csv_line = ','.join(ss.astype(str))
      f.write(csv_line)
      f.write('\n')
    #print(sequence)
