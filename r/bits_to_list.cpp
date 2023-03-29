#include <iostream>
#include <string>


/*
uint64_t string_to_int(std::string s)
{
    int len = s.length();

    uint64_t bits = 0;

    for (int i = 0; i < len; i++)
    {
        char c = s[63-i];
        int cc = (c-48) & 0x1;
        
        bits |= (cc << i);
    }

    return bits;
}

std::string int_to_string(uint64_t d, std::string s)
{
    uint64_t bits = d;

    for (int i = 0; i < 64; i++)
    {
        char c = ((bits >> (63-i)) & 0x1) + 48;
        s[i] = c;
    }
    return s;
}
*/


double str_to_dbl(std::string s)
{
    uint64_t bits = 0;
    for (int i = 0; i < 64; i++)
    {
        uint64_t bit = s[i] - 48;
        bits |= (bit << i);
    }
    return *((double*)(&bits));
}

std::string dbl_to_str(double d, std::string s)
{
    uint64_t b = *((uint64_t*)(&d));
    for (int i = 0; i < 64; i++)
    {
        uint64_t bit = (b >> i) & 0x1;
        s[i] = (char)(bit + 48);
    }
    return s;
}




int main()
{
    double d = 0.2;

    std::string s = "1000000101011100011011000000000101000001010100011000010100111101";
    s = dbl_to_str(d, s);
    std::cout << s << std::endl;

    d = str_to_dbl(s);
    std::cout << d << std::endl;

    return 420;
}