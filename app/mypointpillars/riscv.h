#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <time.h>
#include <stdlib.h>
#include <fcntl.h>
#include <dirent.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <vector>
using namespace std;


int pl_resetn_1(){
  int fd;
  char attr[32];

  DIR *dir = opendir("/sys/class/gpio/gpio172");
  if (!dir) {
    fd = open("/sys/class/gpio/export", O_WRONLY);
    if (fd < 0) {
      perror("open(/sys/class/gpio/export)");
      return -1;
    }
    strcpy(attr, "172");
    write(fd, attr, strlen(attr));
    close(fd);
    dir = opendir("/sys/class/gpio/gpio172");
    if (!dir) {
      return -1;
    }
  }
  closedir(dir);

  fd = open("/sys/class/gpio/gpio172/direction", O_WRONLY);
  if (fd < 0) {
    perror("open(/sys/class/gpio/gpio172/direction)");
    return -1;
  }
  strcpy(attr, "out");
  write(fd, attr, strlen(attr));
  close(fd);

  fd = open("/sys/class/gpio/gpio172/value", O_WRONLY);
  if (fd < 0) {
    perror("open(/sys/class/gpio/gpio172/value)");
    return -1;
  }
  sprintf(attr, "%d", 0);
  write(fd, attr, strlen(attr));

    sprintf(attr, "%d", 1);
  write(fd, attr, strlen(attr));
  close(fd);

  return 0;
}

volatile unsigned int* DMEM_BASE;
volatile unsigned int* IMEM_BASE;
volatile float* DMEM_BASE_FLOAT;
volatile int* DMEM_BASE_INT;
#define DMEM_OFFSET 4096 //later half of DMEM_CONTROL is used data i/o

extern unsigned int riscv_imm(volatile unsigned int *IMEM );

void riscv_init(){
    int uio4_fd = open("/dev/uio4", O_RDWR | O_SYNC);
    DMEM_BASE = (volatile unsigned int*) mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, uio4_fd, 0);
    int uio5_fd = open("/dev/uio5", O_RDWR | O_SYNC);
    IMEM_BASE = (volatile unsigned int*) mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, uio5_fd, 0);

    //write instruction
    riscv_imm(IMEM_BASE);
    DMEM_BASE_FLOAT = (volatile float*) DMEM_BASE;
    DMEM_BASE_INT = (volatile int*) DMEM_BASE;
}

void riscv_run(const vector<float> &points, vector<int> &coors, int num_points, int num_features){
    //assume NDim is 3
    const int MAX_NUM = 682;
    for(int i = 0; i * MAX_NUM < num_points; i++) {
        //one riscv execution
        int n = ((i + 1) * MAX_NUM > num_points) ? (num_points - i * MAX_NUM) : MAX_NUM;
        DMEM_BASE_INT[DMEM_OFFSET + 0] = n;
        for(int j = 0; j < n; j++){
            for(int dim = 0; dim < 3; dim++) {
                DMEM_BASE_FLOAT[DMEM_OFFSET + 1 + j * 3 + dim] = points[(i * MAX_NUM + j) * num_features + dim];
            }
        }
        //set complete flag to false
        DMEM_BASE_INT[DMEM_OFFSET + 4095] = 0;
        //reset to launch RISC-V core
        pl_resetn_1();
        //wait RISC-V execution completion by using polling
        while(1) {
            if (DMEM_BASE_INT[DMEM_OFFSET + 4095] == n) break;
            usleep(1);
        }

        for(int j = 0; j < n; j++){
            for(int dim = 0; dim < 3; dim++){
                coors[(i * MAX_NUM + j) * 3 + dim] = DMEM_BASE_INT[DMEM_OFFSET + 1 + j * 3 + dim + n * 3];
            }
        }
    }
}