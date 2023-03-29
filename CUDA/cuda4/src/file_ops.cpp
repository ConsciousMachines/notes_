#include "file_ops.h"

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <string>


int file_manager::file_exists(const char *filename) 
{
    struct stat buffer;   
    return (stat(filename, &buffer) == 0);
}

void file_manager::write_int_to_init_file(int x)
{
    FILE *my_file = fopen(init_file_name, "wb");
    if (my_file) 
    {
        fwrite(&x, sizeof(int), 1, my_file);
        fclose(my_file);
        printf("wrote %i to init file.\n", x);
    }
    else printf("ERROR OPENING FILE TO WRITE\n");
}

int file_manager::read_int_from_init_file()
{    
    if (file_exists(init_file_name))
    {
        FILE *my_file = fopen(init_file_name, "rb");
        if (my_file)
        {
            int x;
            fread(&x, sizeof(int), 1, my_file);
            fclose(my_file);
            printf("read %i from init file.\n", x);
            return x;
        }
        else 
        {
            printf("ERROR OPENING FILE TO READ\n");
            return -1;
        }
    }
    else // file does not exist so return 0 
    {
        printf("read 0 from init file.\n");
        return 0; 
    }
}