import numpy as np

def preprocssing(filename):
    datasets=[]
    labels=[]
    with open(filename) as R:
        for line in R:
            line=line.strip().split(",")
            l=len(line)
            l=[x for x in line[1:17]+line[18:]+[line[17]]]
            print len(l), l
            datasets.append(l)


    return datasets




def prepare_training_and_testingset(data):
    training_X=[]
    training_Y=[]
    testing_X=[]
    testing_Y=[]
    np.random.shuffle(data)
    l=len(data)
    lm=int(l*.80)
    trainingset=data[:lm]
    testingset=data[lm:]

    training_X=[x[:len(x)-1] for x in trainingset]
    training_Y=[x[-1] for x in trainingset]
    testing_X = [x[:len(x) - 1] for x in testingset]
    testing_Y = [x[-1] for x in testingset]

    return training_X, training_Y, testing_X, testing_Y



def write_file(X1,Y1, X2, Y2):
    with open("training_data.txt",'w') as Tx1, open("training_labels.txt",'w') as Ly1:
        for x1,y1 in zip(X1,Y1):
            Ly1.write(y1+'\n')
            txt=""
            for i in x1:
                txt = txt+i + "\t"
            Tx1.write(txt + '\n')
    with open("testing_data.txt",'w') as Tx1, open("testing_labels.txt",'w') as Ly1:
        for x1,y1 in zip(X2,Y2):
            Ly1.write(y1+'\n')
            txt=""
            for i in x1:
                txt=txt+i+"\t"
            Tx1.write(txt+'\n')













if __name__ == '__main__':
    x1,y1,x2,y2=prepare_training_and_testingset(preprocssing("datasets/parkinsons.txt"))
    write_file(x1,y1,x2,y2)
