#include <iostream>
#include <algorithm>

using namespace std;

#define LOCAL_N 1024
#define RADIUS 3
#define BLOCK_SIZE 16
#define N (2048*2048)
#define THREAD_PER_BLOCK 512

__global__ void mykernel(void) { //funzione eseguita nel device e chiamata dal codice principale

}

__global__ void add(int *a, int *b, int *c) {
	*c = *a + *b;
}

__global__ void vectorAddBlocks(int *a, int *b, int *c) {	//blockIdx.x utilizzato come indice dell'array ogni block geestisce un elemento differente dell'array
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

__global__ void vectorAddThreads(int *a, int *b, int *c) {
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

__global__ void vectorAddBlocksThreads(int *a, int *b, int *c){ //blockDim corrisponde alla grandezza del blocco
	int index = threadIdx.x + blockIdx.x * blockDim.x; //serve per ricavare l'effettivo in indice del'array
	//es 5(indice trhead) + 2(indice blocco) * 8 (Dim di ogni blocco) = 21 indice nell'array)
	c[index] = a[index] + b[index];
}

__global__ void stencil_1d(int *in, int *out) {
	__shared__ int temp[BLOCK_SIZE + 2 * RADIUS]; //dati condivisi e visibili solo tra i threads in uno stesso block
	int gindex = threadIdx.x + blockIdx.x * blockDim.x; //serve per ricavare l'effettivo in indice del'array
	int lindex = threadIdx.x + RADIUS; //indice del vettore temp dove scrivere i dati
	// Read input elements into shared memory
	temp[lindex] = in[gindex];
	if (threadIdx.x < RADIUS) {
		temp[lindex - RADIUS] = in[gindex - RADIUS];
		temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
	}
	// Synchronize (ensure all the data is available) sincronizza tutti i thread di un blocco
	//Usato per prevenire RAW,WAR,WAW hazards
	__syncthreads(); //Come fosse una barriera che tutti i thread devono raggiungere assieme
	// Apply the stencil
	int result = 0;
	for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
		result += temp[lindex + offset];

	// Store the result
	out[gindex] = result;
}

void random_ints (int* a, int n){
	for (int i = 0; i < n; ++i){
		a[i] = rand();
	}
}

void vectorAddBlocksThreads(){
	int *a,*b,*c;				// copie di a b c presenti nel pc
	int *d_a, *d_b, *d_c;		// copie di a b c presenti nella scheda grafica
	int size = N * sizeof(int);

	//alloca spazio per a b c nella scheda grafica
	cudaMalloc((void **)&d_a,size);
	cudaMalloc((void **)&d_b,size);
	cudaMalloc((void **)&d_c,size);

	//Alloco memoria per l'host
	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (int *)malloc(size);

	//inizializzo i vettori
	random_ints(a,N);
	random_ints(b,N);

	//copia i valori nella scheda grafica
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	//lancia add() kernel sulla GPU con N threads
	vectorAddBlocksThreads<<<N/THREAD_PER_BLOCK,THREAD_PER_BLOCK>>>(d_a,d_b,d_c);

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	free(a);
	free(b);
	free(c);

	//Libera memoria sulla GPU
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

}

void vectorAddThreads(){
	int *a,*b,*c;				// copie di a b c presenti nel pc
	int *d_a, *d_b, *d_c;		// copie di a b c presenti nella scheda grafica
	int size = N * sizeof(int);

	//alloca spazio per a b c nella scheda grafica
	cudaMalloc((void **)&d_a,size);
	cudaMalloc((void **)&d_b,size);
	cudaMalloc((void **)&d_c,size);

	//Alloco memoria per l'host
	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (int *)malloc(size);

	//inizializzo i vettori
	random_ints(a,N);
	random_ints(b,N);

	//copia i valori nella scheda grafica
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	//lancia add() kernel sulla GPU con N threads
	vectorAddThreads<<<1,N>>>(d_a,d_b,d_c);

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	free(a);
	free(b);
	free(c);

	//Libera memoria sulla GPU
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

}

void vectorAddBlocks(){
	int *a,*b,*c;				// copie di a b c presenti nel pc
	int *d_a, *d_b, *d_c;		// copie di a b c presenti nella scheda grafica
	int size = N * sizeof(int);

	//alloca spazio per a b c nella scheda grafica
	cudaMalloc((void **)&d_a,size);
	cudaMalloc((void **)&d_b,size);
	cudaMalloc((void **)&d_c,size);

	//Alloco memoria per l'host
	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (int *)malloc(size);

	//inizializzo i vettori
	random_ints(a,N);
	random_ints(b,N);

	//copia i valori nella scheda grafica
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	//lancia add() kernel sulla GPU con N blocks
	vectorAddBlocks<<<N,1>>>(d_a,d_b,d_c);

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	free(a);
	free(b);
	free(c);

	//Libera memoria sulla GPU
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

void simpleAdd(){

	int a,b,c;				// copie di a b c presenti nel pc
	int *d_a, *d_b, *d_c;	// copie di a b c presenti nella scheda grafica
	int size = sizeof(int);

	//alloca spazio per a b c nella scheda grafica
	cudaMalloc((void **)&d_a,size);
	cudaMalloc((void **)&d_b,size);
	cudaMalloc((void **)&d_c,size);

	a = 2;
	b = 7;

	//copia i valori nella scheda grafica
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	//lancia add() kernel sulla GPU
	add<<<1,1>>>(d_a,d_b,d_c);

	// Copy result back to host
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

	//Libera memoria sulla GPU
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	printf("La somma fa: %d\n",c);
}

void stencil(){
	int *in, *out;			// host copies of in, out
	int *d_in, *d_out;		// device copies of in, out
	int size = (LOCAL_N + 2*RADIUS) * sizeof(int); //dimensioni degli array

	// Alloc space for host copies and setup values
	in = (int *)malloc(size);
	random_ints(in, size);
	out = (int *)malloc(size);

	// Alloc space for device copies
	cudaMalloc((void **)&d_in, size);
	cudaMalloc((void **)&d_out, size);

	// Copy to device
	cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);

	// Launch stencil_1d() kernel on GPU
	stencil_1d<<<LOCAL_N/BLOCK_SIZE,BLOCK_SIZE>>>(d_in + RADIUS, d_out + RADIUS);
	// Copy result back to host
	cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

	// Cleanup
	free(in);
	free(out);
	cudaFree(d_in);
	cudaFree(d_out);
}

int main(void) {

	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf(" Device name: %s\n", prop.name);
		printf(" Total Global Memory: %zd\n", prop.totalGlobalMem);
		printf(" Shared Memory Per Block: %zd\n", prop.sharedMemPerBlock);
		printf(" Memory Clock Rate (KHz): %d\n",prop.memoryClockRate);
		printf(" Memory Bus Width (bits): %d\n",prop.memoryBusWidth);
		printf(" Peak Memory Bandwidth (GB/s): %f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth/8)/1.0e6);
		printf(" Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
		printf(" MultiProcessorCount: %d\n\n", prop.multiProcessorCount);
	}

	clock_t t1,t2;
	float diff;
	t1 = clock();
	//Chiamata dal host code al device code detto anche kernel launch
	mykernel<<<1,1>>>();
	printf("Hello world!\n");
	t2 = clock();
	diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
	printf("Tempo %f\n",diff);

	t1 = clock();
	simpleAdd();
	t2 = clock();
	diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
	printf("Tempo simpleAdd: %f\n",diff);

	t1 = clock();
	vectorAddBlocks();
	t2 = clock();
	diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
	printf("Tempo vectorAddBlocks: %f\n",diff);

	t1 = clock();
	vectorAddThreads();
	t2 = clock();
	diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
	printf("Tempo vectorAddThreads: %f\n",diff);

	t1 = clock();
	vectorAddBlocksThreads();
	t2 = clock();
	diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
	printf("Tempo vectorAddBlocksThreads: %f\n",diff);

	t1 = clock();
	stencil();
	t2 = clock();
	diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
	printf("Tempo stencil: %f\n",diff);

	return 0;
}

