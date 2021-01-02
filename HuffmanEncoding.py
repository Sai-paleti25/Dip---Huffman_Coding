import numpy as np
from matplotlib.pyplot import imread 
import heapq as hq
from collections import Counter
%pylab inline
import matplotlib.pyplot as plt
class HeapNode:
    def __init__(self,char,freq):
        self.freq= freq
        self.char=char
        self.left = None
        self.right = None
        
    def __lt__(self, other):
        if other == None:
            return
        else:
            return self.freq < other.freq
    
    def __eq__(self, other):
        if(other == None):
            return False
        if(not isinstance(other, HeapNode)):
            return False
        return self.freq == other.freq
    
    def printTree(self):
        if self.left:
            self.left.printTree()
            print( self.char)
        if self.right:
            self.right.printTree()
class huffmanEncoding:
    def __init__(self):
        self.heap = []
        self.codes = {}
        self.root =[]
        self.dict1={}
    
    img = imread("nature.jpg")
    with open('huffman_shape.txt', 'a') as f:
                print(img.shape, file=f)
    gray_image = np.mean(img,axis=2)
    flatten_image = np.around(gray_image.flatten())
    str1 = " "
    for i in flatten_image:
        val = bin(int(i));
        str1 += val
    with open('Bytes_out1.txt', 'a') as f:
        print(str1, file=f)
    count_frequency = dict(Counter(flatten_image))
    def make_heap(self):
        img = imread("nature.jpg")
        gray_image = np.mean(img,axis=2)
        flatten_image = np.around(gray_image.flatten())
        count_frequency = dict(Counter(flatten_image))
        for key in count_frequency:
            node = HeapNode(key,count_frequency[key])
            hq.heappush(self.heap,node);  
    
    def merge_nodes(self):
        self.root = None
        while len(self.heap)>1:
            node1 = hq.heappop(self.heap)
            if len(self.heap) == 0:
                self.root = node1
            else:
                node2 = hq.heappop(self.heap)
                merged_node = HeapNode(None, node1.freq + node2.freq)
                merged_node.left = node1
                merged_node.right = node2
                hq.heappush(self.heap,merged_node)
    
    
    def make_codes_help(self,node,current_code):
            if(node == None):
                return
            
            if(node.char != None):
                self.codes[node.char] = current_code
                return
            
            self.make_codes_help(node.left,current_code+"0")
            self.make_codes_help(node.right,current_code+"1")

    
    def make_codes(self):
        node = hq.heappop(self.heap)
        self.root = node
        current_code = ""
        print("The current_code is",current_code)
        self.make_codes_help(node,current_code)
            
        
    def get_encoded_text(self, text):
        encoded_text = ""
        
        with open('pixel_codes.txt', 'a') as f:
                print(self.codes, file=f)
        for character in text:
            
            encoded_text += self.codes[character]
       
        return encoded_text
    
    def call_methods(self):
        self.make_heap()
        self.merge_nodes()
        self.make_codes()
        list1 =[]
        img = imread("nature.jpg")
        gray_image = np.mean(img,axis=2)
        flatten_image = np.around(gray_image.flatten())
        encoded_text = self.get_encoded_text(flatten_image)
        with open('encoded_bits.txt', 'a') as f:
                print(encoded_text, file=f)
        return encoded_text
    
    def decodeHuff(self, s):
       #Enter Your Code Here
        temp=self.root
        string=[]
        for i in s:
            c=int(i)
            if c==1:
                temp=temp.right
            elif c==0:
                temp=temp.left
            if temp.right==None and temp.left==None:
                string.append(temp.char)
                temp=self.root
        return string
        
obj = huffmanEncoding()
final_string = obj.call_methods()
output1 = obj.decodeHuff(final_string)
img = imread("nature.jpg")
gray_image = np.mean(img,axis=2)
decoded_image = np.reshape(output1,np.shape(gray_image))
imgplot = plt.imshow(decoded_image,cmap='gray')
plt.show()
