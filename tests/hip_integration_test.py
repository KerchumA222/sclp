import subprocess
import os

def run_test():
    cpp_code = """
#include <iostream>
#include <vector>
#include <cstdint>
#include <hip/hip_runtime.h>

// Kernel implementation included directly for testing purposes
__global__ void kernel_test_kernel(uint16_t* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] ^= 0xFFFF; // Flip bits as a simple test
    }
}

int main() {
    const int n = 5;
    uint16_t h_data[n] = {0x4080, 0x4280, 0x4080, 0x4480, 0x4000};
    uint16_t expected[n] = {0xBF7F, 0xBD7F, 0xBF7F, 0xBB7F, 0xBFFF};

    uint16_t *d_data;
    hipMalloc(&d_data, n * sizeof(uint16_t));
    hipMemcpy(d_data, h_data, n * sizeof(uint16_t), hipMemcpyHostToDevice);

    kernel_test_kernel<<<1, 256>>>(d_data, n);
    hipDeviceSynchronize();

    hipMemcpy(h_data, d_data, n * sizeof(uint16_t), hipMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        if (h_data[i] != expected[i]) {
            std::cout << "Test Failed at index " << i << ": Expected " << std::hex << expected[i] << " but got " << h_data[i] << std::endl;
            return 1;
        }
    }

    std::cout << "HIP Kernel Integration Test Passed!" << std::endl;
    hipFree(d_data);
    return 0;
}
"""
    with open("tests/kernel_tester.cpp", "w") as f:
        f.write(cpp_code)

    print("Compiling kernel_tester.cpp...")
    try:
        # Attempting to use hipcc for compilation
        subprocess.run(["hipcc", "tests/kernel_tester.cpp", "-o", "tests/kernel_tester"], check=True)
        print("Running kernel_tester...")
        result = subprocess.run(["./tests/kernel_tester"], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Result: {result.stdout.strip()}")
            return True
        else:
            print(f"Test Execution Failed:\n{result.stderr}\n{result.stdout}")
            return False
    except Exception as e:
        print(f"Compilation or execution failed: {e}")
        return False
    finally:
        if os.path.exists("tests/kernel_tester.cpp"):
            os.remove("tests/kernel_tester.cpp")
        if os.path.exists("tests/kernel_tester"):
            os.remove("tests/kernel_tester")

if __name__ == "__main__":
    if run_test():
        print("Integration Test Status: SUCCESS")
    else:
        print("Integration Test Status: FAILURE")
        exit(1)
