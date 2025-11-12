#include <stdio.h>

void show_array(int x[], int N)
{
    for (int i=0; i<N; i++)
        printf("x[%d]=%d ", i, x[i]);
    printf("\n");
}


void read_in_numbers(int x[], int N)
{
    printf("Please enter %d numbers now!\n", N);
    for (int i=0; i<N; i++)
    {
        printf("Number %d please:", i);
        scanf("%d", &(x[i]));
    }
}

