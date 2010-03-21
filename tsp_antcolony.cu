/*
 * Copyright 2010 Ruslan Kudubayev.
 */

/* 
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil_inline.h>

// includes, kernels
#include <MersenneTwister_kernel.cu>
#include <tsp_antcolony_kernel.cu>


void printArr(float *data1, int size)
{
  int i,j,k;
  for (j=0; j<size; j++) {
    for (i=0; i<size; i++) {
      k = j*size+i;
      printf("%1.20f ", data1[k]);
    }
    printf("\n");
  }
}

FILE *infile;

int getSize(char* filename)
{
	char line[100];
	int size;
	/* Open the file.  If NULL is returned there was an error */
	if((infile = fopen(filename, "r")) == NULL) {
		printf("Error Opening File.\n");
		exit(1);
	}
	fgets(line, sizeof(line), infile);
	printf(line);
	fgets(line, sizeof(line), infile);
	printf(line);
	fgets(line, sizeof(line), infile);
	printf(line);
	fscanf(infile, "DIMENSION : %d\n", &size);
	printf("%d\n", size);
	fgets(line, sizeof(line), infile);
	printf(line);
	fgets(line, sizeof(line), infile);
	printf(line);
	fgets(line, sizeof(line), infile);
	printf(line);
	return size;
}

void getGraph(int* data, int size) {
	for (int i = 0; i < size*size; ++i) {
        fscanf(infile,"%d", &data[i]);
    }
    fclose(infile);  /* Close the file */
}

void outputfordotformatf(char* filename, float* data, int* path, int size, float scale)
{
	FILE *outfile;
	/* Open the file.  If NULL is returned there was an error */
	if((outfile = fopen(filename, "w")) == NULL) {
		printf("Error Opening File.\n");
		exit(1);
	}
	
	fputs("digraph G{\n",outfile);
	
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) if (i!=j && abs(data[i*size+j])>=1e-16) {
			if (path[i] == j) {
        		fprintf(outfile,"	%d -> %d [label=\"%f\", penwidth=%f, color=red];\n", i, j, data[i*size+j], data[i*size+j]*scale);
			} else {
        		fprintf(outfile,"	%d -> %d [label=\"%f\", penwidth=%f];\n", i, j, data[i*size+j], data[i*size+j]*scale);
        	}
        }
    }
	fputs("}",outfile);
    
    fclose(outfile);  /* Close the file */

}

void outputfordotformati(char* filename, int* data, int* path, int size, float scale)
{
	FILE *outfile;
	/* Open the file.  If NULL is returned there was an error */
	if((outfile = fopen(filename, "w")) == NULL) {
		printf("Error Opening File.\n");
		exit(1);
	}
	
	fputs("digraph G{\n",outfile);
	
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) if (i!=j && data[i*size+j]>0 && data[i*size+j]<1000000) {
			if (path[i] == j) {
        		fprintf(outfile,"	%d -> %d [label=\"%d\", penwidth=%f, color=red];\n", i, j, data[i*size+j], data[i*size+j]*scale);
			} else {
        		fprintf(outfile,"	%d -> %d [label=\"%d\", penwidth=%f];\n", i, j, data[i*size+j], data[i*size+j]*scale);
        	}
        }
    }
	fputs("}",outfile);
    
    fclose(outfile);  /* Close the file */
}

//performs a nearest neighbour simple greedy search to get a value for tau0.
float nearest_neighbour(int* h_A, int size) {
	int cur = 0;
	int visited[WA];
	float res = 0;
	for (int i=0; i<size; i++) visited[i] = 0;
	visited[cur] = 1;
	for (int i=1; i<size; i++) {
		int min = 214748364;
		int minj = i;
		for (int j=0; j<size; j++) if (visited[j] == 0) {
			if (h_A[size*cur+j] < min) {
				min = h_A[size*cur+j];
				minj = j;
			}
		}
		res += min;
		visited[minj] = 1;
		cur = minj;
	}
	res += h_A[size*cur + 0];
	return res;
}


//Align a to nearest higher multiple of b
extern "C" int iAlignUp(int a, int b){
    return ((a % b) != 0) ?  (a - a % b + b) : a;
}

//ceil(a / b)
extern "C" int iDivUp(int a, int b){
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

extern "C" void initMTRef(const char *fname);

void study(float* C, int* A, const int n) {
			int curNode = 0;
			int nNode = -1;
	    	float sum = 0.0f;
	    	float sump = 0.0f;
	    	int visited[WA];
	    	for (int i=0; i<n; i++) visited[i] = -1;
	    	visited[curNode] = 0;
    		// calculate the sum
    		for (int i=0; i<n; ++i) if (visited[i] == -1 && i!=curNode) {
    			// take care of the zero divisions...
    			float eeta;
    			if (A[curNode*n+i]==0) eeta = 1.1f;
    			else eeta = (1.0f/A[curNode*n+i]);
    			sum += C[curNode*n+i] * eeta;
    		}
	    	printf("sum = %1.20f\n", sum);
    		//generate a random number
	    	float ra = (float)rand() / 140009999;
	    	printf("We had r = %1.20f\n", ra);
	    	float target = ra * sum; // precalculate this for the p formula division
    		// calculate the probability and jump if that probability occurs this time.
	    	printf("target = %1.20f\n", target);
    		for (int i=0; i<n; ++i) if (visited[i] == -1 && i!=curNode) {
    			// calculate the probability as per the equation before.
    			float p;
    			float eeta;
    			if (A[curNode*n+i]==0) eeta = 1.1f;
    			else eeta = (1.0f/A[curNode*n+i]);
    			// p calculated here with squaring eeta for better results
    			p = (C[curNode*n+i] * eeta);
    			printf("p = %1.20f, sump = %1.20f\n", p, sump);
    			if (target>sump && target<=p+sump) {
    				// yes. move.
    				nNode = i;
    				break;
    			}
    			nNode = i;
    			sump += p;
    		}
    		printf("We choose %d\n", nNode);
}

////////////////////////////////////////////////////////////////////////////////
// Run test
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv)
{
	char* filenametotest;
	int targetanswer;
	if (argc > 2) {
		filenametotest = argv[1];
		targetanswer = atoi(argv[2]);
	}
    const int n = getSize(filenametotest);
    int blocks = iDivUp(n, BLOCK_SIZE);
    
    int blocks_with_clones = iDivUp(n*CLONES, BLOCK_SIZE);
    
    // allocate host memory for matrice A
    unsigned int size_A = n * n;
    unsigned int mem_size_A = sizeof(int) * size_A;
    int* h_A = (int*) malloc(mem_size_A);
    getGraph(h_A, n);
    
    //if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
    //    cutilDeviceInit(argc, argv);
    //else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );

    // set seed for rand()
    srand(2010);
    
		/*
		This routine is responsible for generating random numbers on the device and leaving
		them there for the other kernel to use.
		*/
   		float *d_Rand;
   		
   		int path_n = (2*n-2) * BLOCK_SIZE * blocks_with_clones;
   		
		int n_per_rng = iAlignUp(iDivUp(path_n, MT_RNG_COUNT), 2);
		int rand_n = MT_RNG_COUNT * n_per_rng;
	
    	printf("Initializing data on the device for %i random samples...\n", path_n);
        cutilSafeCall( cudaMalloc((void **)&d_Rand, rand_n * sizeof(float)) );
        
        //const char *raw_path = cutFindFilePath("MersenneTwister.raw", "");
        //const char *dat_path = cutFindFilePath("MersenneTwister.dat", "");
        initMTRef("MersenneTwister.raw");
        loadMTGPU("MersenneTwister.dat");
        

    // allocate device memory for storing delta matrix
    int* d_A;
    cutilSafeCall(cudaMalloc((void**) &d_A, mem_size_A));
    // copy host memory to device
    cutilSafeCall(cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice) );

    // allocate device memory for tau matrix
    unsigned int size_C = n * n;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* d_C;
    cutilSafeCall(cudaMalloc((void**) &d_C, mem_size_C));
    // allocate host memory for the tau on host
    float* h_C = (float*) malloc(mem_size_C);
    //initialise pheromones tau matrix
    //double sum = 0;
    //for (int i=0; i<size_A; ++i) sum += h_A[i];
    //float h_a_average = sum / (n*n);
    //float tau0 = 1.0f / (h_a_average * (float)n);
    float tau0 = 1.0f/((float)n * nearest_neighbour(h_A, n));  
    printf("tau0: %1.20f\n", tau0);
    for (int i=0; i<size_C; ++i) h_C[i] = tau0;
    // copy host memory to device
    cutilSafeCall(cudaMemcpy(d_C, h_C, mem_size_C,
                              cudaMemcpyHostToDevice) );
    
    // allocate device memory for the path vector
    int size_P = n;
    int mem_size_P = sizeof(int) * size_P;
    int* d_P;
    cutilSafeCall(cudaMalloc((void**) &d_P, mem_size_P));
    // allocate host memory for the R on host
    int* h_P = (int*) malloc(mem_size_P);
    // copy host memory to device
    cutilSafeCall(cudaMemcpy(d_P, h_P, mem_size_P,
                              cudaMemcpyHostToDevice) );
    
    // allocate device memory for best so far
    int* d_best;
    cutilSafeCall(cudaMalloc((void**) &d_best, sizeof(int)));
    // allocate host memory for the best on host
    int* h_best = (int*)malloc(sizeof(int));
    *h_best = 2147483647;
    // copy host memory to device
    cutilSafeCall(cudaMemcpy(d_best, h_best, sizeof(int),
                              cudaMemcpyHostToDevice) ); 
    
    // create and start timer
    unsigned int timer = 0;
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));
    
    // dimensions of the global update kernel
    dim3 threads2(BLOCK_SIDE_UPDATER, BLOCK_SIDE_UPDATER);
    int side_blocks = n/BLOCK_SIDE_UPDATER;
    if (n%BLOCK_SIDE_UPDATER != 0) side_blocks++;
    dim3 grid2(side_blocks, side_blocks);

	// *****************************************************
	// the main block of code that executes the kernel.
	int firsttimeto20 = 0;
	for (int iteration=0; iteration<2048; ++iteration) {
		//generate random numbers for this iteration
		seedMTGPU(rand()%2048);
		RandomGPU<<<32, 128>>>(d_Rand, n_per_rng);
        cutilCheckMsg("RandomGPU() execution failed\n");
        cutilSafeCall( cudaThreadSynchronize() );
	
		float damping = 0.1f;
		
    	// execute the kernel
    	colonise<<< blocks_with_clones, BLOCK_SIZE >>>(d_C, d_A, d_Rand, d_best, d_P, n, damping, tau0);
        cutilSafeCall( cudaThreadSynchronize() );
  	
    	// get the best so far
	    cutilSafeCall(cudaMemcpy(h_best, d_best, sizeof(int),
                              cudaMemcpyDeviceToHost) );
        // record when we get to 20% accuracy
        if ((firsttimeto20 == 0) && (((float)*h_best/targetanswer) <= 1.2f)) {
        	firsttimeto20 = 1;
        	printf("First time to 20 percent accuracy: %f (ms) \n", cutGetTimerValue(timer));
        }
        // if reached the optimal answer then quit, no point to work anymore.
    	if (*h_best == targetanswer) break;
    	
    	// global updating rule here.
    	// can just execute another kernel here which would do that.
    	// the reason is to not copy all the data forth and back but do modifications over there.
    	//if (rand() <= (((float)(iteration%256))/256.0f)) {
    		update_pheromones<<< grid2, threads2 >>>(d_C, d_best, d_P, n, damping);
        	cutilSafeCall( cudaThreadSynchronize() );
        	//blocks = blocks % 8 + 2;
        //}
        
    }
    // ******************************************************  

    // stop and destroy timer
    cutilCheckError(cutStopTimer(timer));
    printf("Processing time: %f (ms) \n", cutGetTimerValue(timer));
    
    cutilCheckError(cutDeleteTimer(timer));

    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed"); 

    // copy result from device to host
    cutilSafeCall(cudaMemcpy(h_C, d_C, mem_size_C,
                              cudaMemcpyDeviceToHost) );
    cutilSafeCall(cudaMemcpy(h_P, d_P, mem_size_P,
                              cudaMemcpyDeviceToHost) );
    cutilSafeCall(cudaMemcpy(h_best, d_best, sizeof(int),
                              cudaMemcpyDeviceToHost) );
                              
    // just looking how this works.
    study(h_C, h_A, n);

    printf("Result: %d\n", *h_best);

    printf("Tau:\n");
    printArr(h_C,n);
    
    outputfordotformati("original_graph.dot",h_A,h_P,n,0.1f);
    outputfordotformatf("tau_graph.dot",h_C,h_P,n,12000.0f);

    // clean up memory
    free(h_A);
    free(h_C);
    free(h_P);
    free(h_best);
    cutilSafeCall(cudaFree(d_A));
    cutilSafeCall(cudaFree(d_C));
    cutilSafeCall(cudaFree(d_Rand));
    cutilSafeCall(cudaFree(d_P));
    cutilSafeCall(cudaFree(d_best));

    cudaThreadExit();
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    runTest(argc, argv);
    cutilExit(argc, argv);
}

