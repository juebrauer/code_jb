#include <iostream>
#include <vector>

template <typename T>
class hkevector
{
    public:

        hkevector()
        {
            capacity = 10;
            nr_elements_stored = 0;
            data = (T*) malloc(capacity * sizeof(T));
        }

        void push_back(T value)
        {
            if (nr_elements_stored == capacity)
            {
                std::cout << "Wir müssen den Speicherbereich größer machen!" << std::endl;
                // Speicherplatz muss größer gemacht werden!
                capacity += 1;
                data = (T*) realloc(data, capacity * sizeof(T));
            }

            data[nr_elements_stored] = value;
            nr_elements_stored++;
        }

        int size()
        {
            return nr_elements_stored;
        }

        T getelement(int i)
        {
            return data[i];
        }

        T& operator[](int i)
        {
            return data[i];
        }

        bool remove(T value)
        {
            // Suche den Wert im Array:
            // Wo steht er?
            for (int i=0; i<nr_elements_stored; i++)
            {
                if (data[i] == value)
                {
                    return remove_idx(i);                    
                }

            }

            // Zu löschender Wert value wurde nicht gefunden!
            // Signalisiere das dem Aufrufer!
            return false;
        }

        bool remove_idx(int i)
        {
            if ((i<0) || (i>nr_elements_stored-1))
            {
                std::cout << "Ungültiger Index!" << std::endl;
                return false;
            }

            // Alles ab Position i+1 nach "oben" schieben
            for (int j=i; j<nr_elements_stored-1; j++)
            {
                data[j] = data[j+1];
            }
            nr_elements_stored--;
            return true;           
        }



    private:

        int capacity;
        int nr_elements_stored;
        T* data; 

    
};


template<typename T>
T f(T x)
{
    return x*x; 
}


int main()
{

    std::cout << f(5) << std::endl;
    
    std::vector<std::string> v;
    v.reserve(5000);
    v.push_back( "Brauer" );
    v.push_back( "Maier" );
    v.push_back( "Schulze" );
    std::cout << "Anzahl der Elemente in v:" << v.size() << std::endl;    
    for (int i=0; i<v.size(); i++)
        std::cout << v[i] << std::endl; 
    
    
    hkevector<double> v2;
    v2.push_back( 1.4783 );
    v2.push_back( 254.543 );
    for (int i=1; i<=5; i++)
        v2.push_back(2+i); 
  
    for (int i=0; i<v2.size(); i++)
        std::cout << v2.getelement(i) << std::endl;
    std::cout << "Anzahl der Elemente in hkevector v2:" << v2.size() << std::endl;
    
    std::cout << std::endl << std::endl;

    v2.remove(6);

    for (int i=0; i<v2.size(); i++)
        std::cout << v2[i] << std::endl; 
    std::cout << "Anzahl der Elemente in hkevector v2:" << v2.size() << std::endl;
   
    
}