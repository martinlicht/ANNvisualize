from math import sqrt
from typing import List, Tuple

import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PIL import Image
from PIL import Image, ImageDraw




# Neural network activation function, say, ReLU
def activation_function( val: float ) -> float:
    if val > 0:
        return val
    else:
        return 0.


# Return a function that computes an artificial neural network
# The ANN has two input neurons and one output neuron, and one hidden layer
def ANN_onelayer( lastvector: List[float], hiddenmatrix: List[List[float]], hiddenvector: List[float] ) -> float:
    
    def actual_function( x: float, y: float ):
        T = len(lastvector)
        ret = 0.
        for t in range(0,T):
            ret += lastvector[t] * activation_function( x * hiddenmatrix[t][0] + y * hiddenmatrix[t][1] + hiddenvector[t] )
        return ret

    return actual_function



# convert a numerical value into a color (for drawing the image) 
def value_to_color( value: float, min_value: float, max_value: float ) -> Tuple[int,int,int]:
    
    if value < min_value: value = min_value 
    if value > max_value: value = max_value

    if max_value == min_value: return (0,0,0)

    if value == max_value: return (255,255,255)
    if value == min_value: return (0,0,0)

    value = ( value - min_value ) / ( max_value - min_value )
    

    thresholds = [      0.,       1./4.,          1./2.,           3./4.,              1. ]
    colors     = [ [0,0,0],   [0,0,255],   [51,255,154],   [255,255,102],   [255,204,204] ]
    # thresholds = [      0.,       1. ]
    # colors     = [ [255,0,0],   [0,0,255] ]
    for i in range(len(colors)-1):
        if ( thresholds[i] <= value ) and ( value <= thresholds[i+1] ):
            l = ( value - thresholds[i] ) / ( thresholds[i+1] - thresholds[i] )
            red   = int( l * colors[i+1][0] + (1-l) * colors[i][0] )
            green = int( l * colors[i+1][1] + (1-l) * colors[i][1] )
            blue  = int( l * colors[i+1][2] + (1-l) * colors[i][2] )
            return (red,green,blue)



# save an image   
def draw_values( filebasename: str, N: int, values: List[List[float]], max_value: float, min_value: float,
                 T: int, hiddenmatrix: List[List[float]], hiddenvector: List[float] ):
    
    ### create the image from those values 

    # create a new blank image 
    image_width  = N
    image_height = N
    block_scale  = 1
    color_default = (0,0,0)
    img = Image.new( 'RGB', (image_width * block_scale, image_height * block_scale), color_default)
    pixels = img.load() 

    # compute the colors
    colors = [ [ value_to_color( values[c][r], min_value, max_value ) for r in range(N+1) ] for c in range(N+1) ]
    
    # fill in the values of that image 

    for r in range(image_height):       # for every row
        for c in range(image_width):    # for every column 
            for p in range(block_scale):
                for q in range(block_scale):
                    pixel_y = r * block_scale + p
                    pixel_x = c * block_scale + q
                    pixels[pixel_x,pixel_y] = colors[c][r]
                    

    for t in range(T):
        
        vx = hiddenmatrix[t][1]
        vy = hiddenmatrix[t][0]
        b  = hiddenvector[t]
        vn2 = (vx*vx+vy*vy)
        vn = sqrt(vn2)
        sx = b * vx/vn2 * image_width/4
        sy = b * vy/vn2 * image_height/4

        cx = image_width/2
        cy = image_height/2
        lx = image_width
        ly = image_height
        
        shape = [ ( cx-sx-lx*vy/vn, cy-sy+ly*vx/vn ), ( cx-sx+lx*vy/vn, cy-sy-ly*vx/vn )]
        img1 = ImageDraw.Draw(img)  
        img1.line(shape,width=1, fill="red" )   
    
    img.save( filebasename + '.gif' )



# plot values
def plot_values( N: int, values: List[List[float]] ):

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    image_width  = N
    image_height = N
    
    x = range(image_width+1)
    y = range(image_height+1)
    X, Y = numpy.meshgrid(x, y)
    #print(X, "Xs <- \n")
    #print(numpy.array( values ), "converted values <- \n")
    ha.plot_surface(X, Y, numpy.array( values ) )

    plt.show()
    



# run example
def run():

    # create an ANN with one-hidden layer
    # T internal neurons, two input and one output neuron
    T = 4
    
    lastvector   = [ numpy.random.normal() for t in range(0,T) ]
    hiddenmatrix = [ [ numpy.random.normal(), numpy.random.normal() ] for t in range(0,T) ]
    hiddenvector = [ numpy.random.normal() for t in range(0,T) ]
    
    myANN = ANN_onelayer( lastvector, hiddenmatrix, hiddenvector )

    # map the values of the ANN over a specific 2D area 
    L = 2
    
    # how many nodal points in each dimension?
    N = 300
    
    # compute the values
    values = [ [ myANN( -L + r * 2*L/N, -L + c * 2*L/N )  for r in range(N+1) ] for c in range(N+1) ]
    
    max_value = max( max( v ) for v in values )
    min_value = min( min( v ) for v in values )

    draw_values( "ann", N, values, max_value, min_value, T, hiddenmatrix, hiddenvector )
    plot_values( N, values )
    
    

run()




