#define REGINT(address) *(volatile int*)(address)
#define REGUINT(address) *(volatile unsigned int*)(address)
#define REGINTPOINT(address) (volatile int*)(address)
#define REGFLOAT(address) *(volatile float*)(address)
#define REGFLOATPOINT(address) (volatile float*)(address)
#define DMEM_BASE  (0xA000C000)

// #include "math.h"
inline int myfloor(float f){
    if (f >= 0) return (int)f;
    else return (int)f - 1;
}

int main(){

    // float a = REGFLOAT(DMEM_BASE);
    // float b = REGFLOAT(DMEM_BASE+4);
    // float c = a + b;
    // REGFLOAT(DMEM_BASE+8) = c;
    int n = REGINT(DMEM_BASE);
    volatile float* points = REGFLOATPOINT(DMEM_BASE + 4);
    volatile int* res = REGINTPOINT(DMEM_BASE + 4 * (n * 3 + 1));
    volatile int* endflg = REGINTPOINT(DMEM_BASE + 4 * 4095);
    const float corrs_range[3] = {-50, -50, -5};
    const float voxel_size[3] = {0.25, 0.25, 8};
    const int grid_size[3] = {400, 400, 1};
    int i;
    int j;
    int j2;
    for(i = 0; i < n; i++) {
        int failed = 0;
        int c[3];
        for(j = 0; j < 3; j++) {
            c[j] = myfloor((points[i * 3 + j] - corrs_range[j]) / voxel_size[j]);
            if ((c[j] < 0 || c[j] >= grid_size[j])) {
                failed = 1;
                break;
            }
        }
        for(j2 = 0; j2 < 3; j2++) {
            if (failed) res[i * 3 + j2] = -1;
            else res[i * 3 + j2] = c[2-j2];
        }
    }
    endflg[0] = n;
    while(1){

    }
    return 1;

}
