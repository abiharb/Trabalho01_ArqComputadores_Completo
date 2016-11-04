/* Código de multiplicação de Matrizes 
Trabalho 01: Arquitetura de Computadores
discente: Maria da Penha de Andrade Abi Harb
Algoritmo: código SEQUENCIAL
Ponto flutuante float
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int main(int argc, char **argv) {
	
	//declaração de variáveis
	float **x, **y, **z;
	int N = 1500;            //tamanho da matriz quadrada
	int numVezes = 100;      //numero de vezes que ocorre a multiplicacao
	int n_threads = 128;    //numero de threads do paralelismo
	int i,j,k;
	int qtS = 0;
	
	//alocando memórias
	x = (float **)malloc (N * sizeof(float *));
	y = (float **)malloc (N * sizeof(float *));
	z = (float **)malloc (N * sizeof(float *));
	
	for (i=0; i<N; i++) {
	  x[i] = (float *)malloc (N * sizeof(float));
	  y[i] = (float *)malloc (N * sizeof(float));
	  z[i] = (float *)malloc (N * sizeof(float));
	}
	
	printf("\n Tamanho Matriz - %d (s)\n", N);
	printf(" Quantidade de vezes - %d (s)\n", numVezes);
	
	printf ("\n ..................... INICIO SEQUENCIAL ..................... \n");
	
	//inicializando as matrizes
	for (i=0; i<N; i++)
	  for (j=0; j<N; j++) {
	    x[i][j] = 1.0;
	    y[i][j] = 0.01;
	    z[i][j] = 0; 
	  }
	
	//Retorna um valor em segundos do tempo decorrido desde algum ponto. 
	double inicioS = omp_get_wtime(); 
	
	//quantidade de vezes que ocorrerá a multiplicação sequencialmente
	for (qtS=0; qtS<numVezes;qtS++) { 
	
		for (i=0; i<N; i++)
		  for (j=0; j<N; j++)
		     for (k=0; k<N; k++)
				z[i][j] += x[i][k] * y[k][j];
	}//código sequencial -Fim
	
	//Retorna um valor em segundos do tempo decorrido desde algum ponto. 
	double fimS = omp_get_wtime();
	printf(" Tempo de execucao total -> %.6f (s)\n", fimS - inicioS);
	
	//printf ("DESALOCANDO VARIÁVEIS DA MEMÓRIA.\n\n");
	free(x);
	free(y);
	free(z);

} //main
