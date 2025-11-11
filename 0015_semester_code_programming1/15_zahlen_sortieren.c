#include <stdio.h>

#define N 5



void show_array(int x[])
{
    for (int i=0; i<N; i++)
        printf("x[%d]=%d ", i, x[i]);
    printf("\n");
}


void read_in_numbers(int x[])
{
    printf("Please enter %d numbers now!\n", N);
    for (int i=0; i<N; i++)
    {
        printf("Number %d please:", i);
        scanf("%d", &(x[i]));
    }
}

void save_array(int x[])
{
    FILE* f;
    f  = fopen("array_sortiert.txt", "w");
    for (int i=0; i<N; i++)
    {
        fprintf(f, "%i: %d\n", i, x[i]);
    }    
    fclose(f);
}


int main()
{
    int x[N];

    read_in_numbers(x);
    show_array(x);

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

        show_array(x);

    } while (swapped == 1);
    
    printf("\nEndergebnis:\n");
    show_array(x);

    save_array(x);
    
}