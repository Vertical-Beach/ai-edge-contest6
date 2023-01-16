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
#define DMEM_OFFSET 4096 //later half of DMEM_CONTROL is used data i/o

extern unsigned int riscv_imm(volatile unsigned int *IMEM );

int main()
{
    int uio0_fd = open("/dev/uio4", O_RDWR | O_SYNC);
    DMEM_BASE = (unsigned int*) mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, uio0_fd, 0);
    int uio1_fd = open("/dev/uio5", O_RDWR | O_SYNC);
    IMEM_BASE = (unsigned int*) mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, uio1_fd, 0);

    //write instruction
    riscv_imm(IMEM_BASE);
    //TEST start
	volatile float* DMEM_BASE_FLOAT = (volatile float*) DMEM_BASE;
	volatile int* DMEM_BASE_INT = (volatile int*) DMEM_BASE;


	//set input data
	int n = 100;
	DMEM_BASE_INT[DMEM_OFFSET + 0] = n;
	for (int i = 0; i < n * 3; i++){
		if (i%3 == 0) DMEM_BASE_FLOAT[DMEM_OFFSET + 1 + i] = 15.5f-1.5f*i;
		if (i%3 == 1) DMEM_BASE_FLOAT[DMEM_OFFSET + 1 + i] = 15.5f-1.5f*i;
		if (i%3 == 2) DMEM_BASE_FLOAT[DMEM_OFFSET + 1 + i] = 2.5f;
	}

	//set complete flag to false
	DMEM_BASE_INT[DMEM_OFFSET + 4095] = 0;

	//reset to launch RISC-V core
	pl_resetn_1();
	//wait RISC-V execution completion by waiting some period or using polling
	// usleep(100);
	while(1) {
		if (DMEM_BASE_INT[DMEM_OFFSET + 4095]) break;
		else usleep(1);
	}
	//get output data
	printf("%d\n", DMEM_BASE_INT[DMEM_OFFSET + 4095]);
	for(int i = 0; i < n * 3; i++) {
		int c = DMEM_BASE_INT[DMEM_OFFSET + 1 + n * 3 + i];
		printf("%d ", c);
		if (i%3 == 2) printf("\n");
	}
    return 0;
}
