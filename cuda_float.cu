/* Código de multiplicação de Matrizes 
Trabalho 01: Arquitetura de Computadores
discente: Maria da Penha de Andrade Abi Harb
Algoritmo: código CUDA
Ponto flutuante float
*/

#define TILE_WIDTH 16   // Definindo tamanho do ladrilho
#define n  500          // Definindo tamanho da matriz
#define vz  1000        // Definindo quantas vezes ocorre as multiplicações

#include<stdio.h>
#include<stdlib.h>
#include <time.h>

void __global__ multiplica(float *Md, float *Nd, float *Pd, int Width);

int main(void)
{	
	//variáveis para contar o tempo
	cudaEvent_t start, stop;
	float time;
  
	float *a,*b,*c;
	
	// adicionar declaração de variaveis vetores das matrizes da GPU
	float *Ga,*Gb,*Gc;
	
	int i;
	
	// variáveis para threads e blocos
	// importante para configurar a execução do processamento e desempenho
	//1024 threads por bloco. Maximo permitido pela configuração da GPU (tanto device 0 ou 1)
	dim3 blocksize(TILE_WIDTH,TILE_WIDTH); 
	dim3 gridsize(n/TILE_WIDTH,n/TILE_WIDTH);
	
	// Alocacao de memoria para as matrizes a,b,c na CPU (host)
	a=(float *)malloc(n*n*sizeof(float ));
	b=(float *)malloc(n*n*sizeof(float ));
	c=(float *)malloc(n*n*sizeof(float ));
	
	// Alocacao de memoria para as matrizes a,b,c na GPU (device)
	cudaMalloc((void **)&Ga,n*n*sizeof(float ));
	cudaMalloc((void **)&Gb,n*n*sizeof(float ));
	cudaMalloc((void **)&Gc,n*n*sizeof(float ));
	
	// Atribuindo valores para as matrizes a e b
	for (i=0;i<n*n;i++) { 
		a[i]=1.0f;     
		b[i]=0.01f;
		c[i]=0; 
	}
	
	printf("\nTamanho do ladrilho = %d\n",TILE_WIDTH);
	printf("Tamanho da matriz = %d\n",n);
	printf("Numero de vezes = %d\n\n",vz);
	
	// Copiando os dados para a GPU
	cudaMemcpy( Ga,a,n*n*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy( Gb,b,n*n*sizeof(int),cudaMemcpyHostToDevice);
	
	//Valor de inicio da contagem do tempo paralelo
	// também foi testado com a função clock()
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0 );

    //Multiplicar 0, 100 e 1000 vezes
    for(int qt = 0; qt<vz; qt++)
		// Precisa informar o numero e hierarquia de threads
		//Função recebe as variáveis da GPU - Função Kernel
		multiplica<<<gridsize,blocksize>>>(Ga,Gb,Gc,n);

	//finalização do tempo
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );

	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	time/=1000.0;
	printf("Tempo de execucao GPU = %f\n\n",time);
	
	//Copiando os dados para a CPU
	cudaMemcpy( c, Gc, n*n*sizeof(float),cudaMemcpyDeviceToHost);

	// Liberando memoria
	free(a);    
	free(b);   
	free(c);   
	
	// Liberando memoria da GPU
	cudaFree(Ga); 
	cudaFree(Gb); 
	cudaFree(Gc);
}

//kernel CUDA que será rodado na GPU 
void __global__ multiplica(float *Md, float *Nd, float *Pd, int Width)
{
	//memoria compartilhada
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
	
	//variaveis do Cuda para Threads
	int bx = blockIdx.x;  
	int by = blockIdx.y;
	
	int tx = threadIdx.x; 
	int ty = threadIdx.y;
	
	// Identificando linha e coluna para a matriz Pd trabalhar
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	
	float Pvalue = 0;
	
	// Loop pra preenchimento da matriz multiplicação
	for (int m = 0; m < Width/TILE_WIDTH; ++m) 
	{
		    // carregando elementos da memoria compartilhada
	        Mds[ty][tx] = Md[Row * Width + (m * TILE_WIDTH + tx)];
	        Nds[ty][tx] = Nd[(m * TILE_WIDTH + ty) * Width + Col];
	        __syncthreads();
	
	        for (int k = 0; k < TILE_WIDTH; ++k)
	            Pvalue += Mds[ty][k] * Nds[k][tx];
	         __syncthreads();
	}
	Pd[Row * Width + Col] = Pvalue;
}


