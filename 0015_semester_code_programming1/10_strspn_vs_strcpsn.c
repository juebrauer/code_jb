#include <stdio.h>
#include <string.h>

int main(void) {
    const char *text = "abc 123xyz";

    size_t len_spn  = strspn(text, "abc 123");
    size_t len_cspn = strcspn(text, "a");

    printf("strspn(text, \"abc\") = %zu\n", len_spn);
    printf("strcspn(text, \"xyz\") = %zu\n", len_cspn);

    return 0;
}
