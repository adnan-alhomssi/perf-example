#include "./PerfEvent.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <ostream>
#include <sys/mman.h>
#include <vector>


void *malloc_huge(size_t size)
{
   void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   madvise(p, size, MADV_HUGEPAGE);
   memset(p, 0, size);
   return p;
}

unsigned count8(int8_t *in, unsigned inCount, int8_t x)
{
   __asm volatile(""); // make sure this is not optimized away when called multiple times
   unsigned count = 0;
   for (unsigned i = 0; i < inCount; i++)
      if (in[i] > x)
         count++;
   return count;
}


unsigned count8SIMD(int8_t *in, unsigned inCount, int8_t x)
{
   __asm volatile(""); // make sure this is not optimized away when called multiple times
   assert(inCount % 32 == 0);
   unsigned count = 0;
   const uint8_t integersPerInstruction = 32; // how many

   __m256i broadcastedRegister = _mm256_set1_epi8(x);

   for (unsigned i = 0; i < inCount; i += integersPerInstruction) { //we jump 64 at a time, because 512/8 = 64 at a time
      __m256i chunk = _mm256_loadu_si256(reinterpret_cast<__m256i const *>(in + i));

      // we can do way better this this poor pop count
      __m256i compare_result = _mm256_cmpgt_epi8(chunk, broadcastedRegister);
      count += _popcnt64(_mm256_extract_epi64(compare_result, 0)) / 8;
      count += _popcnt64(_mm256_extract_epi64(compare_result, 1)) / 8;
      count += _popcnt64(_mm256_extract_epi64(compare_result, 2)) / 8;
      count += _popcnt64(_mm256_extract_epi64(compare_result, 3)) / 8;
   }
   return count;
}

int fib(int x)
{
   if (x == 0)
      return 0;
   else if (x == 1)
      return 1;
   return fib(x - 1) + fib(x - 2);
}

int main()
{
   int32_t inCount = 1024ull * 1024 * 128;
   auto in8 = reinterpret_cast<int8_t *>(malloc_huge(inCount * sizeof(int8_t)));

   for (int32_t i = 0; i < inCount; i++) {
      in8[i] = random() % 100;
   }

   unsigned chunkSize = 32 * 1024; // so it fits completely in L1 cache, try varying the chunkSize

   //verify correctness
   {
      for (auto sel : {1, 10, 50, 90, 99}) {
         assert(count8(in8, inCount, sel) == count8SIMD(in8, inCount, sel));
      }
   }

   PerfEvent e;
   for (auto sel : {1, 10, 50, 90, 99}) {
      e.setParam("name", "scalar  8");
      e.setParam("selectivity", sel);
      // second parameter is scale, all the results except time, IPC,  CPU,  GHz are normalized by this scale
      PerfEventBlock b(e, inCount);

      unsigned chunkCount = (inCount * sizeof(uint8_t)) / chunkSize;
      for (unsigned i = 0; i < chunkCount; i++)
         count8(in8, inCount / chunkCount, sel);
   }

   for (auto sel : {1, 10, 50, 90, 99}) {
      e.setParam("name", "SIMD  8");
      e.setParam("selectivity", sel);
      PerfEventBlock b(e, inCount);

      unsigned chunkCount = (inCount * sizeof(uint8_t)) / chunkSize;
      for (unsigned i = 0; i < chunkCount; i++)
         count8SIMD(in8, inCount / chunkCount, sel);
   }

   {
      for (auto until : {10, 40}) {
         e.setParam("name", "Fib");
         e.setParam("until", until);
         PerfEventBlock b(e, 1);
         fib(until);
      }
   }

   return 0;
}