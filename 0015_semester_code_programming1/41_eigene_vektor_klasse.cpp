#include <iostream>
#include <vector>


class hkevector
{
    public:

        hkevector()
        {
            capacity = 10;
            nr_elements_stored = 0;
            data = (int*) malloc(capacity * sizeof(int));
        }

        void push_back(int value)
        {
            if (nr_elements_stored == capacity)
            {
                std::cout << "Wir müssen den Speicherbereich größer machen!" << std::endl;
                // Speicherplatz muss größer gemacht werden!
                capacity += 1;
                data = (int*) realloc(data, capacity * sizeof(int));
            }

            data[nr_elements_stored] = value;
            nr_elements_stored++;
        }

        int size()
        {
            return nr_elements_stored;
        }

        int getelement(int i)
        {
            return data[i];
        }



    private:

        int capacity;
        int nr_elements_stored;
        int* data; 

    
};

int main()
{

    std::vector<int> v;
    v.reserve(5000);
    v.push_back( 100 );
    v.push_back( 200 );
    for (int i=1; i<=5; i++)
        v.push_back(200+i*100);         
    std::cout << "Anzahl der Elemente in v:" << v.size() << std::endl;    
    for (int i=0; i<v.size(); i++)
        std::cout << v[i] << std::endl; 
 
    
    hkevector v2;
    v2.push_back( 100 );
    v2.push_back( 200 );
    for (int i=1; i<=15; i++)
        v2.push_back(200+i*100); 
    std::cout << "Anzahl der Elemente in hkevector v2:" << v2.size() << std::endl;
    for (int i=0; i<v2.size(); i++)
        std::cout << v2.getelement(i) << std::endl; 
    
}