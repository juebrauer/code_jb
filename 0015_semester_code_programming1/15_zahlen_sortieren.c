#include <stdio.h>

#define N 5

int x[N];

void show_array()
{
    for (int i=0; i<N; i++)
        printf("x[%d]=%d ", i, x[i]);
    printf("\n");
}


void read_in_numbers()
{
    printf("Please enter %d numbers now!\n", N);
    for (int i=0; i<N; i++)
    {
        printf("Number %d please:", i);
        scanf("%d", &(x[i]));
    }
}


int main()
{
    read_in_numbers();
    show_array();

    int swapped;
    do
    {
        swapped = 0;

        for (int j=0; j<N-1; j++)
        if (x[j] > x[j+1])
        {
            int tmp;
            tmp = x[j];    // tmp = a
            x[j] = x[j+1]; // a = b
            x[j+1] = tmp;  // b = tmp

            swapped = 1;
        }

        show_array();

    } while (swapped == 1);
    
    printf("\nEndergebnis:\n");
    show_array();
    
}