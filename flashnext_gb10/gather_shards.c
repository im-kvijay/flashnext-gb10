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
