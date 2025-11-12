#include <stdio.h>

#define N 5

void show_array(int x[], int s);
void read_in_numbers(int x[], int s);
void save_array(int x[], int s);

int main()
{
    int x[N];

    read_in_numbers(x, N);
    show_array(x, N);

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

        show_array(x, N);

    } while (swapped == 1);
    
    printf("\nEndergebnis:\n");
    show_array(x, N);

    save_array(x, N);
    
}