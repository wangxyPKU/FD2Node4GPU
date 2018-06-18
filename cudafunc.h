
extern "C" {
    void GetDeviceName();
    void GpuCalculate(float *fai, int H, int W, int my_rank, int comm_sz);
}
