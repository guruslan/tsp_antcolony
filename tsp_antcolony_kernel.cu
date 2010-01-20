/*
 * Copyright 2009 Ruslan Kudubayev.
 */

/*
 * Device code.
 */

#ifndef _TSP_ANTCOLONY_KERNEL_H_
#define _TSP_ANTCOLONY_KERNEL_H_

#include <stdio.h>
#include "tsp_antcolony.h"

////////////////////////////////////////////////////////////////////////////////
// colonisation
////////////////////////////////////////////////////////////////////////////////
__global__ void colonise(float* C, int* A, int* Cell, int* d_best, int* d_path, const int n, const int start, const float R) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // obtain what node we start from
    int startNode = (start+bx+by)%n;
    int curNode =  startNode;
    int visited[WA];
    for (int i=0; i<n; ++i) visited[i] = -1;
    visited[startNode] = 0;
    int cost = 0.0;
    int collected;
    
    //coordinates for the random cell
    int randc = tx+ty;
    
    // in a while loop do movements according to the matrix updating tau
    for (collected=1; collected<n; ++collected) {
   	 	int nNode = -1;
    	// get random
	    Cell[randc] = (Cell[randc] * 1103515245 + 12345);
    	float ra = (float)((Cell[randc]/65536) % 32768)/32768;
    	//exploit or explore
    	if (ra<=0.2) { //exploit the paths
    		float max = -1.0;
    		for (int i=0; i<n; ++i) if (visited[i]==-1 && i!=curNode) {
    			if (C[curNode*n+i] > max) {
    				max = C[curNode*n+i];
    				nNode = i;
    			}
    		}
		} else { // explore properly
	    	float sum = 0.0;
	    	float sump = 0.0;
    		// calculate the sum
    		for (int i=0; i<n; ++i) if (visited[i] == -1 && i!=curNode) {
    			sum += (1.0/A[curNode*n+i]) * (C[curNode*n+i]);
    		}
    		//generate a random number
	    	Cell[randc] = (Cell[randc] * 1103515245 + 12345);
    		ra = (float)((Cell[randc]/65536) % 32768)/32768;
    		//float ra = ((float)tx*ty)/(BLOCK_SIZE*BLOCK_SIZE);
    		// calculate the probability and jump if that probability occurs this time.
    		for (int i=0; i<n; ++i) if (visited[i] == -1 && i!=curNode) {
    			// calculate the probability as per the equation before.
    			float p = ((1.0/A[curNode*n+i]) * (C[curNode*n+i])) / sum;
    			if (ra<=p+sump) {
    				// yes. move.
    				nNode = i;
    				break;
    			}
    			nNode = i;
    			sump += p;
    		}
    	}
    	if (nNode >= 0) {
    		cost = cost + A[curNode*n+nNode];
    		visited[curNode] = nNode;
    		curNode = nNode;
    	} else {
    		// don't really prefer to go there
    		break;
    	}
    }
    
    if (collected == n) {
    	cost = cost + A[curNode*n+startNode];
    	visited[curNode] = startNode;
    
    	// so if we found a suitable route:
    	// update feromones.
    	for (int i=0; i<n; ++i) {
    		for (int j=0; j<n; ++j) {
    			C[i*n+j] *= (1 - R);
    		}
    	}
    	for (int i=0; i<n; ++i) {
    		int k = visited[i];
    		C[i*n+k] += (1.0/cost);
        }
	    // after done, evaluate the path with the best achieved so far
   		if (cost < *d_best) {
    		*d_best = cost;
    		for (int i=0; i<n; ++i) d_path[i] = visited[i];
    	}
    }
}

#endif // #ifndef _TSP_ANTCOLONY_KERNEL_H_
