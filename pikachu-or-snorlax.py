import numpy as np
from PIL import Image
import sys
from tqdm import tqdm

def sigmoid(x):
   if x<0:
       return np.exp(x)/(1+np.exp(x))
   else:
       return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return x * (1-x)
    
def think(example,weights,bias):
    maxm = 0
    k = 0
    a = 0
    neither = 0
    b = int(sigmoid(np.dot(example,weights)+bias)[0])==1
    for i in range(len(example)):
        if(maxm<example[i]):
            maxm = example[i]
            k = i
    if(k == 0):
        neither = 0
    elif(k == 1):
        neither = 0
    else:
        neither = 1
        print("Image is of neither.")
    if neither == 0:
       if int(sigmoid(np.dot(example,weights)+bias)[0])==1:
          print("Image is of Pikachu.")
       else:
          print("Image is of Snorlax.")
    np.save('weights.npy' , weights )
    pressenter = input("")
    return sigmoid(np.dot(example,weights)+bias)
    
def weights_gen(x,y):
   l =[]
   for i in range(x * y):
       l.append(np.random.randn())
   return(np.array(l).reshape(x, y))

#Does the neural training
def go(example,training_inputs,labels,bias=np.random.rand(1)):
    print("Working...")
    labels = labels.reshape(training_inputs.shape[0],training_inputs.ndim -1)
    try:
        weights = np.load('weights.npy')
    except:
        weights = weights_gen(training_inputs.shape[1],training_inputs.ndim -1)
    weights = np.array(weights,dtype = float)
    for epoch in tqdm(range(4000)):
        inputs = training_inputs
        x = np.dot(training_inputs,weights) + bias
        for c,i in enumerate(x):
            prediction = sigmoid(i)
            error = prediction - labels
            cost = error
            derivative = sigmoid_der(prediction)
            slope = cost * derivative
        inputs = training_inputs.T
        weights = weights - 0.2 *np.dot(inputs,slope)
        for num in slope:
            bias = bias - (0.2 * num)
    result = think(example,weights,bias)
    return result
        
def input_data(path1,path2,path3,path11,path22,path33,pathex):
    print("go brr")
    #open and resize all images to 512x512
    newsize = (512, 512)
    
    pikachu1 = Image.open(path1)
    pikachu1 = pikachu1.resize(newsize)

    pikachu2 = Image.open(path2)
    pikachu2 = pikachu2.resize(newsize)

    pikachu3 = Image.open(path3)
    pikachu3 = pikachu3.resize(newsize)

    snorlax1 = Image.open(path11)
    snorlax1 = snorlax1.resize(newsize)

    snorlax2 = Image.open(path22)
    snorlax2 = snorlax2.resize(newsize)

    snorlax3 = Image.open(path33)
    snorlax3 = snorlax3.resize(newsize)

    example = Image.open(pathex)
    example = example.resize(newsize)

    
    #convert to list of lists from list of tuples
    rpikachu1 = list(pikachu1.getdata())
    rpikachu2 = list(pikachu2.getdata())
    rpikachu3 = list(pikachu3.getdata())
    rsnorlax1 = list(snorlax1.getdata())
    rsnorlax2 = list(snorlax2.getdata())
    rsnorlax3 = list(snorlax3.getdata())
    rex = list(example.getdata())
    
    #flatten the lists and convert to np array
    fp1 = np.array([x for sets in rpikachu1 for x in sets])
    fp2 = np.array([x for sets in rpikachu2 for x in sets])
    fp3 = np.array([x for sets in rpikachu3 for x in sets])
    fs1 = np.array([x for sets in rsnorlax1 for x in sets])
    fs2 = np.array([x for sets in rsnorlax2 for x in sets])
    fs3 = np.array([x for sets in rsnorlax3 for x in sets])
    fex = np.array([x for sets in rex for x in sets])

    #The "correct" outputs to the training data
    labels = np.array([[1,1,1,0,0,0]])

    #training data - 3 pictures of pikachu, 3 pictures of snorlax
    training_data = np.array([fp1,fp2,fp3,fs1,fs2,fs3],dtype = float)

    #calls function which does the neural training
    res = go(fex,training_data,labels)
    
    

#paths to the image files
path1 = "pikachu1.jpg"
path2 = "pikachu2.jpg"
path3 = "pikachu3.jpg"
path11 = "snorlax1.jpg"
path22 = "snorlax2.jpg"
path33 = "snorlax3.jpg"
#user inputs path of image to be analyzed
pathex = sys.argv[1]
print(pathex)

#call main function input_data which takes the image paths as parameters
input_data(path1,path2,path3,path11,path22,path33,pathex)



