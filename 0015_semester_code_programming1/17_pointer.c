#include <stdio.h>

int main()
{   
    int arr[3] = {1, 2, 3};
    int *p = arr;
    printf("Das Array geht los im Speicher an Stelle %p\n", p);

    p = p + 1;
    printf("p=%p\n", p);

    p = p + 1;
    printf("p=%p\n", p);

    printf("%d", *p);

    printf("Das 3. Element von Array arr steht im Speicher hier: %p\n", &(arr[2]));
}