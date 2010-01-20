/*
 * Copyright 2009 Ruslan Kudubayev.
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
#include <tsp_antcolony_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
void randomInit(float*, int);
void printDiff(float*, float*, int, int);

extern "C"
void computeGold(float*, const float*, unsigned int);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    runTest(argc, argv);

    cutilExit(argc, argv);
}

void printArr(float *data1, int size)
{
  int i,j,k;
  for (j=0; j<size; j++) {
    for (i=0; i<size; i++) {
      k = j*size+i;
      printf("%f ", data1[k]);
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
    // allocate host memory for matrice A
    unsigned int size_A = n * n;
    unsigned int mem_size_A = sizeof(int) * size_A;
    int* h_A = (int*) malloc(mem_size_A);
    getGraph(h_A, n);
    
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );

    // set seed for rand()
    srand(2006);


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
    for (int i=0; i<size_C; ++i) h_C[i] = 1.0;
    // copy host memory to device
    cutilSafeCall(cudaMemcpy(d_C, h_C, mem_size_C,
                              cudaMemcpyHostToDevice) );
                              
    // allocate device memory for the awesome random matrix
    unsigned int size_R = BLOCK_SIZE;
    unsigned int mem_size_R = sizeof(int) * size_R;
    int* d_R;
    cutilSafeCall(cudaMalloc((void**) &d_R, mem_size_R));
    // allocate host memory for the R on host
    int* h_R = (int*) malloc(mem_size_R);
    for (int i=0; i<size_R; ++i) h_R[i] = (i+1)*i*i + (rand()*10000);
    // copy host memory to device
    cutilSafeCall(cudaMemcpy(d_R, h_R, mem_size_R,
                              cudaMemcpyHostToDevice) );
    
    // allocate device memory for the awesome path vector
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

    // setup execution parameters
    dim3 threads(BLOCK_SIZE, 1);
    dim3 grid(BLOCKS, 1);

	int firsttimeto20 = 0;
	for (int iteration=0; iteration<2500; ++iteration) {
		float damping = 0.1;
		if (iteration%200 == 199) damping += 0.3;
		int start = iteration % n;
    	// execute the kernel
    	colonise<<< grid, threads >>>(d_C, d_A, d_R, d_best, d_P, n, start, damping);
	    cutilSafeCall(cudaMemcpy(h_best, d_best, sizeof(float),
                              cudaMemcpyDeviceToHost) );
        if ((firsttimeto20 == 0) && (((float)*h_best/targetanswer) <= 1.2f)) {
        	firsttimeto20 = 1;
        	printf("First time to 20 percent accuracy: %f (ms) \n", cutGetTimerValue(timer));
        }
    	if (*h_best == targetanswer) break;
    }

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


    // compute reference solution
    //float* reference = (float*) malloc(mem_size_C);
    //computeGold(reference, h_A, WA);

    // check result
    //CUTBoolean res = cutCompareL2fe(reference, h_C, size_C, 1e-6f);
    //printf("Test %s \n", (1 == res) ? "PASSED" : "FAILED");
    //if (res!=1) printDiff(reference, h_C, WA);
    printf("Result: %d\n", *h_best);

    printf("Tau:\n");
    printArr(h_C,n);
    
    outputfordotformati("original_graph.dot",h_A,h_P,n,0.1);
    outputfordotformatf("tau_graph.dot",h_C,h_P,n,n*n-20);

    // clean up memory
    free(h_A);
    free(h_C);
    free(h_R);
    free(h_P);
    free(h_best);
    //free(reference);
    cutilSafeCall(cudaFree(d_A));
    cutilSafeCall(cudaFree(d_C));
    cutilSafeCall(cudaFree(d_R));
    cutilSafeCall(cudaFree(d_P));
    cutilSafeCall(cudaFree(d_best));

    cudaThreadExit();
}
