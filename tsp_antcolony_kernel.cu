/*
 * Copyright 2010 Ruslan Kudubayev.
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
__global__ void colonise(float* C, int* A, float* Rand, int* d_best, int* d_path, const int n, const float R, const float tau0) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int antid = ((bx+by)*BLOCK_SIZE+(tx+ty)) % n;
    int cloneid = ((bx+by)*BLOCK_SIZE+(tx+ty)) / n;
    
    //calculate the offset in the Rand array
    int offset = (antid*CLONES + cloneid)*(2*n-2);
    
    float ra;
    
    int startNode = (antid+cloneid)%n;
    int curNode =  startNode;
    int visited[WA];
    for (int i=0; i<n; ++i) visited[i] = -1;
    visited[startNode] = 0;
    int cost = 0;
    int collected;
    
    // in a while loop do movements according to the matrix updating tau
    for (collected=1; collected<n; ++collected) {
   	 	int nNode = -1;
    	// get random
	    ra = Rand[offset + collected*2 - 2];
    	//exploit or explore
    	if (ra > 0.2f) { //exploit the paths, get the max one simply
    		float max = -1.0f;
    		int first = 1;
    		for (int i=0; i<n; ++i) if (visited[i]==-1 && i!=curNode) {
    			if (first || C[curNode*n+i] > max) {
    				max = C[curNode*n+i];
    				nNode = i;
    				first = 0;
    			}
    		}
		} else { // explore properly
	    	float sum = 0.0f;
	    	float sump = 0.0f;
    		// calculate the sum
    		for (int i=0; i<n; ++i) if (visited[i] == -1 && i!=curNode) {
    			// take care of the zero divisions...
    			float eeta;
    			if (A[curNode*n+i]==0) eeta = 1.1f;
    			else eeta = (1.0f/A[curNode*n+i]);
    			sum += C[curNode*n+i] * eeta;
    		}
    		//generate a random number
	    	ra = Rand[offset + collected*2 - 1];
	    	float target = ra * sum; // precalculate this for the p formula division
    		// calculate the probability and jump if that probability occurs this time.
    		for (int i=0; i<n; ++i) if (visited[i] == -1 && i!=curNode) {
    			// calculate the probability as per the equation before.
    			float p;
    			float eeta;
    			if (A[curNode*n+i]==0) eeta = 1.1f;
    			else eeta = (1.0f/A[curNode*n+i]);
    			// p calculated here with squaring eeta for better results
    			p = (C[curNode*n+i] * eeta);
    			if (target>sump && target<=p+sump) {
    				// yes. move.
    				nNode = i;
    				break;
    			}
    			nNode = i;
    			sump += p;
    		}
    	}
    	if (nNode >= 0) {
    		// accept the next node
    		cost = cost + A[curNode*n+nNode];
    		visited[curNode] = nNode;
	    	// apply local updating rule right now.
			C[curNode*n+nNode] = (1.0f - R) * C[curNode*n+nNode] + (R * tau0);
			// move on
    		curNode = nNode;
    	} else {
    		// don't really prefer to go there
    		// this means that an ant has arrived
    		// into some deadlock where it is best to die than 
    		// lead anyone else here.
    		break;
    	}
    	
    }
    
    if (collected == n) {
    	cost = cost + A[curNode*n+startNode];
    	visited[curNode] = startNode;
    	C[curNode*n+startNode] = (1.0f - R) * C[curNode*n+startNode] + (R * tau0);
    
	    // after done, evaluate the path with the best achieved so far
   		if (cost < *d_best) {
    		*d_best = cost;
    		for (int i=0; i<n; ++i) d_path[i] = visited[i];
    	}
    }
}

////////////////////////////////////////////////////////////////////////////////
// global update
////////////////////////////////////////////////////////////////////////////////
__global__ void update_pheromones(float* C, int* d_best, int* d_path, const int n, const float A) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int i = (bx * BLOCK_SIDE_UPDATER) + tx;
    int j = (by * BLOCK_SIDE_UPDATER) + ty;
    
    if (i<n && j<n) {
	    // update feromones.
	    float deposition = 0.0f;
	    float evaporation = C[i*n+j] * (1.0f - A);
	    
	    if (d_path[i] == j) {
   			deposition = A/(*d_best);
   			C[i*n+j] = evaporation+deposition;
    	}
    }
}
#endif // #ifndef _TSP_ANTCOLONY_KERNEL_H_
