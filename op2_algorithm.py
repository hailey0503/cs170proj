import math
import time
import random
import datetime
from node import Node
from node import length
from node import total_length
from node import dump
import matplotlib.pyplot as plt
import numpy


l_min = None
pic = 0
# v = [-100,4100,-100,2100]

def framec( solution, nc ):
    global pic
    cities = [ (n.x,n.y) for n in solution ]
    cities = numpy.array( cities )
    plt.axis( [-100,4100,-100,2100] )
    plt.axis('off')
    plt.plot(cities[:,0],cities[:,1],'ko')
    plt.title('{} Cities, 5 Search Algorithms'.format(nc))
    plt.savefig( ("%05d" % pic)+'.png')
    plt.clf()
    print pic
    pic += 1

def frame0(solution, nodes, l, title):
    global pic
    cities = [(n.x,n.y) for n in solution]
    cities.append(( solution[0].x, solution[0].y ))
    cities = numpy.array( cities )
    all_node = [(n.x,n.y) for n in nodes]
    all_nodes = numpy.array(all_node)

    plt.axis([-100, 4100, -100, 2100])
    plt.axis('off')
    plt.plot(cities[:,0], cities[:,1], 'bo-')
    plt.plot(all_nodes[:,0], all_nodes[:,1], 'ko')
    plt.title('{}  Tour length {:.1f}'.format(title,l))
    plt.savefig( ("%05d" % pic)+'.png')
    plt.clf()
    print pic
    pic += 1

nn = 0
def frame(nodes, solution, sn, t, c, y, x, z, gain):
    global pic
    global nn

    cities = [(n.x,n.y) for n in solution]
    cities = numpy.array(cities)

    cities2 = [(c.x,c.y), (y.x,y.y)]
    cities3 = [(x.x,x.y), (z.x,z.y)]
    cities2 = numpy.array(cities2)
    cities3 = numpy.array(cities3)

    plt.plot(cities[:,0],cities[:,1],'bo-')
    #plt.scatter(cities[:,0], cities[:,1],s=50,c='k')

    if gain < 0:
        plt.scatter(cities2[:,0], cities2[:,1], c='r',s=180)
        plt.plot(cities2[:,0],cities2[:,1], c='r',linewidth=2)
        plt.scatter(cities3[:,0], cities3[:,1],c='b',s=150)
        plt.plot(cities3[:,0],cities3[:,1], c='r',linewidth=2)

    else:
        plt.scatter(cities2[:,0], cities2[:,1], c='g',s=180)
        plt.plot(cities2[:,0], cities2[:,1], c='g',linewidth=2)
        plt.scatter(cities3[:,0], cities3[:,1], c='b',s=150)
        plt.plot(cities3[:,0], cities3[:,1], c='g',linewidth=2)

    plt.axis( [-100,4100,-100,2100] )
    plt.axis('off')

    # In first few frames there might not be an l_min yet
    if l_min is None:
        plt.title('(4)  SA Temp {:4.1f} Best Tour ---\nSwaps {}  Gain {:12.2f} '.format(t, l_min, nn, gain))
    else:
        plt.title('(4)  SA Temp {:4.1f} Best Tour {:6.1f}\nSwaps {}  Gain {:12.2f} '.format(t, l_min, nn, gain))

    plt.savefig( ("%05d" % pic)+'.png')
    plt.clf()
    pic += 1
    print pic


def frame4(nodes, solution, sn, c, y, x, z, gain):
    global pic
    global nn

    l_min = total_length( nodes, solution )
    cities = [ (n.x,n.y) for n in solution ]
    cities = numpy.array( cities )

    cities2 = [ (c.x,c.y), (y.x,y.y) ]
    cities3 = [ (x.x,x.y), (z.x,z.y) ]
    cities2 = numpy.array( cities2 )
    cities3 = numpy.array( cities3 )

    plt.plot(cities[:,0],cities[:,1],'bo-')
    #plt.scatter(cities[:,0], cities[:,1],s=50,c='k')

    if gain < 0:
        plt.scatter(cities2[:,0], cities2[:,1],c='r',s=180)
        plt.plot(cities2[:,0],cities2[:,1],c='r',linewidth=2)
        plt.scatter(cities3[:,0], cities3[:,1],c='b',s=150)
        plt.plot(cities3[:,0],cities3[:,1],c='r',linewidth=2)

    else:
        plt.scatter(cities2[:,0], cities2[:,1],c='g',s=180)
        plt.plot(cities2[:,0],cities2[:,1],c='g',linewidth=2)
        plt.scatter(cities3[:,0], cities3[:,1],c='b',s=150)
        plt.plot(cities3[:,0],cities3[:,1],c='g',linewidth=2)

    plt.axis( [-100,4100,-100,2100] )
    plt.axis('off')

    plt.title('(3)  2-Opt Tour {:6.1f}'.format(l_min))
    plt.savefig( ("%05d" % pic)+'.png')
    plt.clf()
    pic += 1
    print pic






def optimize2opt(nodes, solution, number_of_nodes):
    best = 0
    best_move = None
    # For all combinations of the nodes
    for ci in range(0, number_of_nodes):
        for xi in range(0, number_of_nodes):
            yi = (ci + 1) % number_of_nodes  # C is the node before Y
            zi = (xi + 1) % number_of_nodes  # Z is the node after X

            c = solution[ ci ]
            y = solution[ yi ]
            x = solution[ xi ]
            z = solution[ zi ]
            # Compute the lengths of the four edges.
            cy = length( c, y )
            xz = length( x, z )
            cx = length( c, x )
            yz = length( y, z )

            # Only makes sense if all nodes are distinct
            if xi != ci and xi != yi:
                # What will be the reduction in length.
                gain = (cy + xz) - (cx + yz)
                # Is is any better then best one sofar?
                if gain > best:
                    # Yup, remember the nodes involved
                    best_move = (ci,yi,xi,zi)
                    best = gain

    print best_move, best
    if best_move is not None:
        (ci,yi,xi,zi) = best_move
        # This four are needed for the animation later on.
        c = solution[ ci ]
        y = solution[ yi ]
        x = solution[ xi ]
        z = solution[ zi ]

        # Create an empty solution
        new_solution = range(0,number_of_nodes)
        # In the new solution C is the first node.
        # This we we only need two copy loops instead of three.
        new_solution[0] = solution[ci]

        n = 1
        # Copy all nodes between X and Y including X and Y
        # in reverse direction to the new solution
        while xi != yi:
            new_solution[n] = solution[xi]
            n = n + 1
            xi = (xi-1)%number_of_nodes
        new_solution[n] = solution[yi]

        n = n + 1
        # Copy all the nodes between Z and C in normal direction.
        while zi != ci:
            new_solution[n] = solution[zi]
            n = n + 1
            zi = (zi+1)%number_of_nodes
        # Create a new animation frame
        frame4(nodes, new_solution, number_of_nodes, c, y, x, z, gain)
        return (True,new_solution)
    else:
        return (False,solution)

