# Opens pdsit file of the dataset
import csv
reader = csv.reader(open("pdist.csv", "rt"), delimiter=",")
x = list(reader)
pdist = np.array(x).astype("float")

# Finds the average pairwise distance between adjacent points for each element in sequence
def find_pdist_average(seq):

  average_pdist = []
  placeholder = []

  for i in range(len(seq)):
    placeholder.append(pdist[seq[i][0]][seq[i][1]])
    placeholder.append(pdist[seq[i][0]][seq[i][2]])
    placeholder.append(pdist[seq[i][1]][seq[i][2]])
    average_pdist.append(np.mean(placeholder))
    placeholder = []

  return average_pdist

#
good_seq_pdist = find_pdist_average(good_seq)
plt.title('Good sequence average pdist')
plt.xlabel('Distance value')
plt.ylabel('Sequence')
plt.hist(good_seq_pdist)
plt.show()
print(np.mean(good_seq_pdist))

bad_seq_pdist = find_pdist_average(bad_seq)
plt.title('Bad sequence average pdist')
plt.xlabel('Distance value')
plt.ylabel('Sequence')
plt.hist(bad_seq_pdist)
plt.show()
print(np.mean(bad_seq_pdist))

plt.title('Good sequences vs Bad sequences average pdist')
plt.xlabel('Distance value')
plt.ylabel('Sequence')
plt.hist(bad_seq_pdist)
good_seq_pdist = find_pdist_average(good_seq)
plt.hist(good_seq_pdist)
plt.show()
