#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* All pointers remain owned by live CPU tensors in CheckpointShards. */
void flashnext_gather_shards(const uintptr_t *sources, const int64_t *starts,
                            const int64_t *ends, int64_t shards,
                            const int64_t *ids, int64_t count, int64_t valid_rows,
                            size_t width, uint8_t *output, int threads) {
    #pragma omp parallel for num_threads(threads) if(count >= 64 && threads > 1) schedule(static)
    for (int64_t i = 0; i < count; ++i) {
        int64_t id = ids[i];
        uint8_t *destination = output + (size_t)i * width;
        if (id < 0 || id >= valid_rows) {
            memset(destination, 0, width);
            continue;
        }
        int64_t lo = 0, hi = shards;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            if (starts[mid] <= id) lo = mid + 1;
            else hi = mid;
        }
        int64_t shard = lo - 1;
        if (shard < 0 || id >= ends[shard]) {
            memset(destination, 0, width);
            continue;
        }
        const uint8_t *source = (const uint8_t *)sources[shard];
        memcpy(destination, source + (size_t)(id - starts[shard]) * width, width);
    }
}

/* Same rows read with pread at high queue depth (O_DIRECT when aligned=1). Page-fault gathers
   through a file mapping reach only a few concurrent device reads; one step's
   hundreds of random rows then stall the decoder. fds/bases give each part's
   descriptor and file offset. Returns the number of failed reads. */
#include <stdlib.h>
#include <unistd.h>
int flashnext_pread_shards(const int *fds, const int64_t *bases, const int64_t *starts,
                           const int64_t *ends, int64_t shards, const int64_t *ids,
                           int64_t count, int64_t valid_rows, size_t width,
                           uint8_t *output, int threads, int aligned) {
    int failures = 0;
    #pragma omp parallel num_threads(threads) reduction(+:failures)
    {
        uint8_t *bounce = NULL;
        size_t capacity = ((width + 4095) / 4096 + 1) * 4096;
        if (posix_memalign((void **)&bounce, 4096, capacity)) bounce = NULL;
        #pragma omp for schedule(dynamic, 1)
        for (int64_t i = 0; i < count; ++i) {
            int64_t id = ids[i];
            uint8_t *destination = output + (size_t)i * width;
            int64_t lo = 0, hi = shards;
            while (lo < hi) {
                int64_t mid = lo + (hi - lo) / 2;
                if (starts[mid] <= id) lo = mid + 1;
                else hi = mid;
            }
            int64_t shard = lo - 1;
            if (id < 0 || id >= valid_rows || shard < 0 || id >= ends[shard]) {
                memset(destination, 0, width);
                continue;
            }
            int64_t offset = bases[shard] + (id - starts[shard]) * (int64_t)width;
            if (!aligned) {
                /* Buffered: page-cache hits are copies, misses read in parallel. */
                if (pread(fds[shard], destination, width, offset) != (ssize_t)width) failures++;
                continue;
            }
            int64_t first = offset & ~(int64_t)4095;
            size_t span = (size_t)(((offset + (int64_t)width + 4095) & ~(int64_t)4095) - first);
            /* The last aligned block may extend past end of file: a short
               read is valid when it covers this row. */
            ssize_t got = bounce ? pread(fds[shard], bounce, span, first) : -1;
            if (got < (ssize_t)(offset - first + (int64_t)width)) {
                failures++;
                continue;
            }
            memcpy(destination, bounce + (offset - first), width);
        }
        free(bounce);
    }
    return failures;
}
