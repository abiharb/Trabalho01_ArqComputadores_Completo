/* Código de multiplicação de Matrizes 
Trabalho 01: Arquitetura de Computadores
discente: Maria da Penha de Andrade Abi Harb
Algoritmo: código Paralelo com OPENMP
Ponto flutuante double
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int main(int argc, char **argv) {
	
	//declaração de variáveis
	double **x, **y, **z;
	int N = 1500;           //tamanho da matriz quadrada
	int numVezes = 100;    //numero de vezes que ocorre a multiplicacao
	int n_threads = 128;    //numero de threads do paralelismo
	int i,j,k;
	int qt = 0;
	
	//alocando memória
	x = (double **)malloc (N * sizeof(double *));
	y = (double **)malloc (N * sizeof(double *));
	z = (double **)malloc (N * sizeof(double *));
	
	for (i=0; i<N; i++) {
	  x[i] = (double *)malloc (N * sizeof(double));
	  y[i] = (double *)malloc (N * sizeof(double));
	  z[i] = (double *)malloc (N * sizeof(double));
	}
	
	printf("\n Tamanho Matriz - %d (s)\n", N);
	printf(" Quantidade de vezes - %d (s)\n", numVezes);
	printf(" Numero de threads - %d (s)\n", n_threads);
	
	//inicializando as matrizes
	for (i=0; i<N; i++)
	  for (j=0; j<N; j++) {
	     x[i][j] = 1.0;
	     y[i][j] = 0.01;
	     z[i][j] = 0; 
	  }
	
	printf ("\n..................... INICIO PARALELO ..................... \n\n");
	//quantidade de vezes que ocorrerá a multiplicação paralelamente
	
	//Retorna que um valor em segundos do tempo decorrido desde algum ponto. 
	double inicioPar = omp_get_wtime(); 
	
	for (qt=0; qt<numVezes;qt++) { 
	
		/* Região paralelismo para multiplicar matriz 
		shared: Especifica que uma ou mais variáveis devem ser compartilhadas entre todos os threads
		private: cada segmento deve ter sua própria instância de uma variável */    
		#pragma omp parallel shared (x,y,z) private (i,j,k) num_threads(n_threads)
		{
		
		  //SCHEDULE permite uma variedade de opções por especificar quais iterações dos laços são executadas por quais threads
		  //DYNAMIC divide o espaço de iteração em pedaços de tamanho chunksize , e os atribui para as threads com uma política first-come-first-served
		  #pragma omp for schedule (dynamic)
		  for (i=0; i<N; i++)
		    for (j=0; j<N; j++)
		       for (k=0; k<N; k++)
			  z[i][j] += x[i][k] * y[k][j];
		} 
	} // for - quantidade de vezes que ocorrerá a multiplicação
	
	//Retorna que um valor em segundos do tempo decorrido desde algum ponto. 
	double fimPar = omp_get_wtime();
	printf(" Tempo total paralel-> %.6f (s)\n", fimPar - inicioPar); 
	
	//printf ("DESALOCANDO VARIÁVEIS DA MEMÓRIA.\n\n");
	free(x);
	free(y);
	free(z);

} //main
